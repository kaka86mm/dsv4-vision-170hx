# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import errno
import ctypes
import fcntl
import json
import mmap
import os
import platform
import time
from collections.abc import Callable

import numpy as np
import torch

from vllm.distributed.device_communicators.shm_broadcast import (
    check_shm_free_space,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


# Backing store for the shared region. "shm" (default) is a file in /dev/shm;
# "memfd" is an anonymous memfd published to sibling processes through
# /proc/<pid>/fd/<n>, which sidesteps a small /dev/shm (containers often cap it
# well below RAM) -- the region is then bounded only by the memory cgroup.
_REGION_BACKEND = os.environ.get("VLLM_KV_OFFLOAD_REGION_BACKEND", "shm")


def _memfd_create(name: str) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    libc.memfd_create.restype = ctypes.c_int
    fd = libc.memfd_create(name.encode(), 0)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"memfd_create failed: {os.strerror(err)}")
    return fd


def _reclaim_stale_memfd_markers(exclude_path: str) -> int:
    """memfd regions die with their last fd holder, so only the rendezvous
    marker in /dev/shm can leak; reap markers nobody holds a lock on."""
    reclaimed = 0
    try:
        names = os.listdir("/dev/shm")
    except OSError:
        return 0
    for name in names:
        if not (name.startswith("vllm_offload_") and name.endswith(".memfd")):
            continue
        path = os.path.join("/dev/shm", name)
        if path == exclude_path:
            continue
        try:
            if time.time() - os.stat(path).st_mtime < 60.0:
                continue
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue
            for q in (path, path + ".meta"):
                try:
                    os.unlink(q)
                except OSError:
                    pass
            reclaimed += 1
            logger.warning("Reclaimed stale KV offload memfd marker %s", path)
        finally:
            os.close(fd)
    return reclaimed


def _reclaim_stale_regions(exclude_path: str) -> int:
    """Delete offload region files whose owning engine is gone.

    Every process holds a shared flock on its region file for the file's
    whole lifetime (taken right after open, fd kept open until cleanup),
    and the kernel drops flocks automatically on process death --
    including SIGKILL. A region file we can lock exclusively therefore
    has no live owner. Without this reaper, every fail-fast boot crash
    leaks its region into /dev/shm (engine ids are fresh per boot) until
    the free-space precheck refuses to boot at all, turning a transient
    crash into a permanent crash loop that restart policies cannot heal.

    Files younger than 60s are skipped: a booting sibling engine's
    workers open+lock their creator's file within moments, and the grace
    period keeps the reaper away from that window. Regions created by
    builds that predate flocking are indistinguishable from stale ones;
    they are only safe to reap because deployments stop the old engine
    before starting a new build.
    """
    reclaimed = 0
    try:
        names = os.listdir("/dev/shm")
    except OSError:
        return 0
    for name in names:
        if not (name.startswith("vllm_offload_") and name.endswith(".mmap")):
            continue
        path = os.path.join("/dev/shm", name)
        if path == exclude_path:
            continue
        try:
            if time.time() - os.stat(path).st_mtime < 60.0:
                continue
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue  # lock held -> a live engine owns this region
            for p in (path, path + ".meta"):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            reclaimed += 1
            logger.warning(
                "Reclaimed stale KV offload region %s (no live owner holds "
                "its lock)",
                path,
            )
        finally:
            os.close(fd)
    return reclaimed



def _interleave_across_numa_nodes(mm: mmap.mmap, length: int) -> None:
    """mbind(MPOL_INTERLEAVE) the region across all online NUMA nodes.

    Under the kernel's default first-touch policy every populated page
    lands on the toucher's node, and all workers run next to their GPUs
    on one socket. A region approaching that node's capacity starves it:
    observed on a 2-node 2 TB box, 888 GiB of a 1 TiB region landed on
    node0 (1009 GiB), leaving ~10 GiB free there, and the driver's
    node-local allocations during cudaHostRegister then failed with
    cudaErrorMemoryAllocation - a boot crash loop. Interleaving spreads
    the footprint evenly; remote-socket DMA is slower but this is a
    staging tier. Must run BEFORE population: mbind only affects pages
    that are not yet allocated.
    """
    knob = os.environ.get("VLLM_KV_OFFLOAD_NUMA_INTERLEAVE", "auto")
    if knob == "0":
        return
    if platform.machine() != "x86_64":
        return
    try:
        online = open("/sys/devices/system/node/online").read().strip()
    except OSError:
        return
    nodes: set[int] = set()
    for part in online.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            nodes.update(range(int(lo), int(hi) + 1))
        elif part:
            nodes.add(int(part))
    if len(nodes) < 2 or max(nodes) >= 64:
        return
    if knob != "1":
        # auto: interleave ONLY when the region is big enough to threaten a
        # single node. Small regions are better off with first-touch - each
        # rank's slots land on its GPU-local node and transfers stay on-socket.
        # Threshold: half the smallest node's capacity.
        min_node_bytes = None
        for n in nodes:
            try:
                with open(f"/sys/devices/system/node/node{n}/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            b = int(line.split()[-2]) * 1024
                            if min_node_bytes is None or b < min_node_bytes:
                                min_node_bytes = b
                            break
            except OSError:
                return
        if min_node_bytes is None or length < min_node_bytes // 2:
            logger.info(
                "Offload region (%.0f GB) fits comfortably in one NUMA node; "
                "keeping first-touch placement for transfer locality "
                "(set VLLM_KV_OFFLOAD_NUMA_INTERLEAVE=1 to force interleave)",
                length / 1e9,
            )
            return
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    mask = 0
    for n in nodes:
        mask |= 1 << n
    nodemask = (ctypes.c_ulong * 1)(mask)
    buf = ctypes.c_char.from_buffer(mm)
    try:
        addr = ctypes.addressof(buf)
        _SYS_MBIND = 237  # x86_64
        _MPOL_INTERLEAVE = 3
        ret = libc.syscall(
            _SYS_MBIND,
            ctypes.c_void_p(addr),
            ctypes.c_size_t(length),
            ctypes.c_int(_MPOL_INTERLEAVE),
            nodemask,
            ctypes.c_ulong(max(nodes) + 2),
            ctypes.c_uint(0),
        )
    finally:
        del buf
    if ret != 0:
        logger.warning(
            "mbind(MPOL_INTERLEAVE) failed (errno=%d); offload region pages "
            "will follow first-touch NUMA placement",
            ctypes.get_errno(),
        )
    else:
        logger.info(
            "Offload region interleaved across NUMA nodes %s", sorted(nodes)
        )



def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


def _madvise_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    mmap_obj.madvise(_MADV_POPULATE_WRITE, offset, length)


def _fallback_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    # Touch one byte per page via a read-modify-write so existing bytes are
    # preserved — a peer worker may have already written KV data into this
    # shared mmap by the time we run on a kernel without MADV_POPULATE_WRITE.
    arr = np.frombuffer(mmap_obj, dtype=np.uint8)
    arr[offset : offset + length : mmap.PAGESIZE] |= 0


def _get_populate_write_fn(
    mmap_obj: mmap.mmap,
) -> Callable[[mmap.mmap, int, int], None]:
    """Select the pre-faulting method once for this mmap."""
    try:
        _madvise_populate_write(mmap_obj, 0, mmap.PAGESIZE)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        logger.warning(
            "MADV_POPULATE_WRITE is not supported; falling back to per-page "
            "writes for mmap pre-population. Startup may be slower."
        )
        return _fallback_populate_write
    return _madvise_populate_write


class SharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride

        self.backend = _REGION_BACKEND
        if self.backend not in ("shm", "memfd"):
            raise ValueError(
                f"VLLM_KV_OFFLOAD_REGION_BACKEND={self.backend!r}; use shm or memfd"
            )
        suffix = "memfd" if self.backend == "memfd" else "mmap"
        # For memfd this is only the rendezvous marker (a few bytes); the
        # region itself is anonymous memory reachable via meta["memfd"].
        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.{suffix}"
        self._marker_fd: int | None = None
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each block row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        if self.backend == "memfd":
            self._init_memfd(engine_id)
        else:
            self._init_shm()
        self._map_and_populate(rank, num_blocks, cpu_page_size)

    def _init_memfd(self, engine_id: str) -> None:
        meta_path = self.mmap_path + ".meta"
        try:
            self._marker_fd = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            fcntl.flock(self._marker_fd, fcntl.LOCK_SH)
            _reclaim_stale_memfd_markers(self.mmap_path)
            self.fd: int | None = _memfd_create(f"vllm_offload_{engine_id}")
            os.ftruncate(self.fd, self.total_size_bytes)
            meta_tmp = meta_path + ".tmp"
            with open(meta_tmp, "w") as f:
                json.dump(
                    {
                        "num_blocks": self.num_blocks,
                        "row_stride": self._row_stride,
                        "total_size_bytes": self.total_size_bytes,
                        "memfd": f"/proc/{os.getpid()}/fd/{self.fd}",
                    },
                    f,
                )
            os.replace(meta_tmp, meta_path)
            self._creator = True
            logger.info(
                "Created memfd offload region %s (%.2f GB), marker %s",
                f"/proc/{os.getpid()}/fd/{self.fd}",
                self.total_size_bytes / 1e9,
                self.mmap_path,
            )
        except FileExistsError:
            self._marker_fd = os.open(self.mmap_path, os.O_RDWR)
            fcntl.flock(self._marker_fd, fcntl.LOCK_SH)
            meta = self._wait_for_meta(meta_path)
            memfd_path = meta.pop("memfd", None)
            self._check_geometry(meta)
            if not memfd_path:
                raise RuntimeError(f"{meta_path} carries no memfd path")
            deadline = time.monotonic() + 30.0
            while True:
                try:
                    self.fd = os.open(memfd_path, os.O_RDWR)
                    break
                except OSError as e:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Cannot open the creator's memfd {memfd_path}: {e}"
                        ) from e
                    time.sleep(0.05)
            _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.info("Opened existing memfd offload region %s", memfd_path)

    def _wait_for_meta(self, meta_path: str) -> dict:
        deadline = time.monotonic() + 30.0
        while not os.path.exists(meta_path):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for {meta_path}; the region "
                    "creator did not publish its geometry."
                )
            time.sleep(0.05)
        with open(meta_path) as f:
            return json.load(f)

    def _check_geometry(self, meta: dict) -> None:
        expected = {
            "num_blocks": self.num_blocks,
            "row_stride": self._row_stride,
            "total_size_bytes": self.total_size_bytes,
        }
        if meta != expected:
            raise RuntimeError(
                "Shared KV offload region geometry mismatch: creator "
                f"published {meta} but this process computed {expected}. "
                "All workers and the scheduler must derive one geometry "
                "(see KVCacheConfig.max_worker_kv_bytes_per_block); with "
                "pipeline parallelism a per-stage mismatch here would "
                "silently corrupt offloaded KV."
            )

    def _init_shm(self) -> None:
        try:
            # Exclusive create — only one worker succeeds
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            # Advertise liveness: hold a shared lock on the region file for
            # as long as this process lives (fd stays open until cleanup).
            # The reaper below only deletes region files nobody holds a
            # lock on, so this is what protects OUR region from a sibling
            # engine's reaper.
            fcntl.flock(self.fd, fcntl.LOCK_SH)
            # Reap regions leaked by dead engines (fail-fast crashes are
            # SIGKILLed and cannot clean up after themselves) BEFORE the
            # free-space check, so a crash loop frees its own garbage and
            # the restart policy can actually self-heal.
            _reclaim_stale_regions(self.mmap_path)
            # Fail fast with an actionable message if /dev/shm cannot hold the
            # region. Otherwise page allocation fails later as an inscrutable
            # EFAULT/SIGBUS. Stale regions from SIGKILLed engines (e.g.
            # docker force-recreate) are the usual culprit.
            vfs = os.statvfs("/dev/shm")
            free = vfs.f_bavail * vfs.f_frsize
            if free < self.total_size_bytes:
                stale = [
                    f
                    for f in os.listdir("/dev/shm")
                    if f.startswith("vllm_offload_") and f != os.path.basename(self.mmap_path)
                ]
                os.close(self.fd)
                os.unlink(self.mmap_path)
                raise RuntimeError(
                    f"/dev/shm has {free / 1e9:.1f} GB free but the KV offload "
                    f"region needs {self.total_size_bytes / 1e9:.1f} GB "
                    "(after reaping stale regions). Other offload regions "
                    f"present: {stale or 'none'} - these are live (lock held) "
                    "or too young to reap. Free /dev/shm space, increase its "
                    "size, or set VLLM_KV_OFFLOAD_REGION_BACKEND=memfd."
                )
            os.ftruncate(self.fd, self.total_size_bytes)
            # Publish the region geometry so late-joining processes can verify
            # they agree. PP stages computing different geometries used to
            # race on this file: the loser waited forever for a size the
            # winner never set, or worse, mapped overlapping incompatible
            # layouts. Written atomically so openers never see a partial file.
            meta_tmp = self.mmap_path + ".meta.tmp"
            with open(meta_tmp, "w") as f:
                json.dump(
                    {
                        "num_blocks": self.num_blocks,
                        "row_stride": self._row_stride,
                        "total_size_bytes": self.total_size_bytes,
                    },
                    f,
                )
            os.replace(meta_tmp, self.mmap_path + ".meta")
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            # Same liveness lock as the creator: joiners keep the region
            # alive too (the creator process alone may die first).
            fcntl.flock(self.fd, fcntl.LOCK_SH)
            self._check_geometry(self._wait_for_meta(self.mmap_path + ".meta"))
            _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.info("Opened existing mmap file %s", self.mmap_path)

    def _map_and_populate(self, rank: int | None, num_blocks: int, cpu_page_size: int) -> None:
        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # Forbid transparent huge pages on this mapping. khugepaged collapses
        # neighbouring 4K pages into 2M pages asynchronously; a huge page
        # spanning two ranks' slot boundaries makes their per-slot
        # cudaHostRegister ranges overlap at the physical-page level, and the
        # cross-process pin/DMA-map churn during early warmup surfaces as a
        # probabilistic async cudaErrorInvalidValue (boot-time crash race).
        _MADV_NOHUGEPAGE = getattr(mmap, "MADV_NOHUGEPAGE", 15)
        try:
            self.mmap_obj.madvise(_MADV_NOHUGEPAGE)
        except OSError:
            logger.warning("MADV_NOHUGEPAGE not supported; leaving THP enabled")

        # Spread a near-node-sized region across NUMA nodes before any page
        # is touched (fork feature; auto-gated inside).
        _interleave_across_numa_nodes(self.mmap_obj, self.total_size_bytes)

        populate_write_fn = _get_populate_write_fn(self.mmap_obj)

        if rank is not None:
            # Populate only this worker's pages (one slot per block row).
            worker_offset = rank * cpu_page_size
            _t0 = time.perf_counter()
            page_size = self.page_size
            for block in range(num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                populate_write_fn(self.mmap_obj, aligned_offset, aligned_length)
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
                num_blocks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            populate_write_fn(self.mmap_obj, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self._canonical_offset = 0
        self.is_pinned: bool = False

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_blocks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per block for this  tensor.
        """
        assert self.rank is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_next_canonical_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view shared by all workers for one
        canonical tensor (canonical layout).

        Must be called once per canonical tensor, instead of
        create_next_worker_view. The full mmap layout is:

            |<-------- canonical area ------->|<-------- unused ------->|
            |  all workers share this area    |                         |
            |                                 |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            ^                ^
            _canonical_offset=0, then advances by each tensor's size

        Each canonical_t{i} cell is that tensor's canonical page for the
        block. Canonical areas are carved consecutively from the start of
        each block row; consecutive rows are separated by row_stride. Every
        worker gets the identical byte ranges and writes only its disjoint
        bytes within them, as described by its canonical mappings — unlike
        create_next_worker_view, which gives each worker a private
        cpu_page_size slot per row.

        The trailing unused bytes exist only when the canonical pages sum to
        less than row_stride: page-alignment padding of the row, or
        deduplication of KV replicated across workers (e.g. the MLA latent),
        where one canonical copy replaces world_size worker copies.

        Args:
            tensor_page_size: Canonical bytes per block for this tensor.
        """
        new_offset = self._canonical_offset + tensor_page_size
        assert new_offset <= self._row_stride
        view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._canonical_offset,
        )
        self._canonical_offset = new_offset
        self._views.append(view)
        return view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        # Do NOT cudaHostUnregister here: with per-slot pinning that is tens
        # of thousands of driver calls taking minutes, during which a crashed
        # engine looks hung -- the container never exits, so the docker
        # restart policy cannot self-heal a boot-time failure. The driver
        # releases every pin automatically when the process exits, which is
        # exactly where cleanup() runs.
        if self.is_pinned and self._base is not None:
            self._pinned_slot_offsets = []
            self.is_pinned = False
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._marker_fd is not None:
            try:
                os.close(self._marker_fd)
            except Exception:
                logger.warning("Failed to close marker fd", exc_info=True)
            self._marker_fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                try:
                    os.unlink(self.mmap_path + ".meta")
                except OSError:
                    pass
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
