# Use-after-free from CUDA-graph memory profiling double-init (upstream a4d70bef3 / PR #53306)

Prepared 2026-08-26. All line numbers refer to upstream `299ebd094` unless noted.
Status: root cause confirmed by controlled experiment matrix on our deployment;
final full-revival validation (default compile + piecewise graphs + fix) running at time of writing.

## TL;DR

`profile_cudagraph_memory()` (introduced by a4d70bef3, "[Model Runner V2] Reserve
CUDA graph memory", PR #53306) runs `initialize_kv_cache` **twice**: once against a
throwaway minimal KV cache that is captured against and then freed, and once for
real. Lazily-initialized state that caches **raw device pointers** during the
throwaway pass is not reset by `_teardown_profiling_state`, so the first real
warmup forward touches freed memory → async illegal memory access (IMA), detected
at the next stream sync/enqueue point (in our PP deployments: NCCL isend on the
first pipeline stage).

Hybrid GDN models (Qwen3.5 family, GDN linear attention + full attention) crash
**deterministically at the first warmup prefill** whenever `cudagraph_mode != NONE`.
Two aggravating factors:

1. The profiling pass runs **even when its result is discarded**:
   `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` gates only whether the estimate is
   *applied* (`gpu_worker.py:556-561`), not whether the pass *executes*
   (`gpu_worker.py:548-553`). Everyone with graphs enabled pays the double-init.
2. `_teardown_profiling_state` (`vllm/v1/worker/gpu/cudagraph_utils.py:852`) clears
   `kv_caches` / `attn_groups` / `kv_cache_config` / `cudagraph_manager` /
   per-layer `kv_cache`, but **not** lazily-initialized structures that captured
   `data_ptr()`s of the throwaway tensors.

## Environment

- Model: Qwen3.8-27B AWQ (arch `Qwen3_5ForConditionalGeneration`), hybrid
  GDN linear attention + full attention, `--kv-cache-dtype int8_per_token_head`,
  1M yarn context, KV offloading connector configured (crash reproduces with and
  without it).
- Hardware: 8× RTX A6000 (sm86), driver 595.84, CUDA 13.0 image, torch 2.13.0.
- Topologies: reproduces on TP8PP1, TP4PP2, TP2PP4, DP2TP4 alike.
- Crash is deterministic (≈15 consecutive boots across configs).

## Crash signature

Last log line is always the model warmup entering its first real prefill
(`warmup_kernels` → `worker_execute_model(prefill_output)`); then every rank hits:

```
misc/strongstream.cc:426 (ncclStrongStreamSynchronize) NCCL WARN Cuda failure
'an illegal memory access was encountered'
... torch.distributed.DistBackendError: NCCL error ... ncclUnhandledCudaError
```

or, in PP runs, all PP0 workers die inside `isend` (`parallel_state.py:1067`) —
i.e. the IMA is **asynchronous** and surfaces at the next enqueue/sync point,
not at the faulting kernel.

`compute-sanitizer --tool memcheck` reports **zero** violations before the crash
window (memcheck serializes launches and degrades capture, so the poisoned path
is not exercised) — consistent with a dangling-pointer/异步 mechanism rather than
a static OOB index.

## The experiment matrix (the conviction)

The profiling pass executes iff `cudagraph_mode != NONE` (`gpu_worker.py:549-552`).
That predicate matches our stability matrix **exactly**:

| # | Config | cudagraph_mode | profiling runs? | Result |
|---|--------|----------------|------------------|--------|
| 1 | `--enforce-eager` (+MTP) | NONE | no | **stable** (full benchmark passes) |
| 2 | `-O0` (CompilationMode NONE) | NONE | no | **stable** (4-topology benchmark suite passes) |
| 3 | default compile (piecewise graphs) | ≠NONE | yes | IMA at first warmup prefill |
| 4 | default compile, no MTP | ≠NONE | yes | IMA (same) |
| 5 | default compile, `--mamba-cache-mode` none vs `align` | ≠NONE | yes | IMA (both) |
| 6 | default compile, KV offload removed | ≠NONE | yes | IMA (same) |
| 7 | default compile, `VLLM_GDN_DECODE_KERNEL=triton` | ≠NONE | yes | IMA (same) |
| 8 | `mode 0` + `cudagraph_mode FULL` | ≠NONE | yes | worker dies silently after capture |
| 9 | `mode 0` + `cudagraph_mode FULL_DECODE_ONLY` | ≠NONE | yes | IMA |
| 10 | **inductor fully ON + `cudagraph_mode NONE`** | NONE | no | **stable** ← decisive |
| 11 | old base `62195e978` (predates a4d70bef3), default compile | — | pass does not exist | **stable** (production, weeks) |

Row 10 is the decisive split: with full inductor codegen enabled and only the
graph mode forced to NONE, the boot sails through the exact step that kills
rows 3-9. Inductor codegen is innocent; the profiling flow is the trigger.

Rows 4-7 exclude the other candidates we chased first (speculative decoding /
vllm-project/vllm#40756 draft-buffer race — a separate real issue, fence ported
independently; mamba align state migration; the KV offloading connector; the new
fused CUDA GDN decode kernel).

## Mechanism (code-level)

1. `gpu_worker.py:548-553`: after `profile_run()`, if `cuda_alike` and
   `cudagraph_mode != NONE`, call `model_runner.profile_cudagraph_memory()` —
   note **no check of `envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`** here; the
   env is consulted only at `gpu_worker.py:556-561` to decide whether the
   estimate is *used*.
2. `cudagraph_utils.py:734` `profile_cudagraph_memory`:
   `_init_minimal_kv_cache_for_profiling` allocates a minimal KV cache,
   `initialize_kv_cache(minimal_config, is_profiling=True)`, runs
   `capture_model()` dummy forwards to sample graph memory, then
   `_teardown_profiling_state(runner)` frees everything; the real
   `initialize_kv_cache` runs afterwards with fresh (different) allocations.
3. During the profiling dummy forwards, lazily-initialized ("once-guarded")
   structures observe the **throwaway** tensors and cache raw addresses:
   - (align mode) mamba spec-decode context:
     `model_states/mamba_hybrid.py:139/156` — created once
     (`if self._mamba_ctx is None` / `if not ctx.is_initialized`), and
     `initialize_from_forward_context` stores **`state_base_addrs` — raw
     `data_ptr()`s of the state tensors** plus block-table pointers, with an
     explicit "captured once, stable data_ptr" comment
     (`mamba_utils.py:862/873`).
   - (align-independent; rows 4-7 crash without align) at least one further
     structure of the same class. Primary candidate per a4d70bef3's **own
     docstring**, which acknowledges that "inductor graph partition reclaims the
     storages of earlier cudagraph recordings ... leading to use-after-free
     crashes": the commit tears down its `CUDAGraphWrapper`s, but inductor
     cudagraph-tree static input bindings created during the profiling capture
     are not on the cleanup list. Secondary candidates: persistent GDN decode
     workspaces, any attention-backend scratch keyed to first-seen kv tensors.
4. `_teardown_profiling_state` (`cudagraph_utils.py:852`) frees the throwaway
   tensors but resets none of the above; the once-guards see "already
   initialized" and keep the stale pointers.
5. First real warmup forward dereferences freed device memory (e.g.
   `tl.load(state_base_addrs_ptr + ...)` style raw-pointer access) → async IMA →
   surfaced at the next NCCL enqueue (PP isend / allreduce strongstream sync).

Why non-hybrid models mostly survive: dense attention paths re-read
`layer.kv_cache` (which teardown *does* reset and real init repopulates); the
failure needs a component that snapshots raw addresses once. Hybrid GDN state
management is exactly that component, which is why Qwen3.5/GDN (and likely
Qwen3-Next / K3-style hybrids) are the canaries.

## Minimal fix (safe, restores pre-a4d70bef3 behavior when estimate unused)

Gate the *execution* on the same opt flag that gates the *use* (our fork commit
`0437a1671`):

```diff
         cudagraph_memory_estimate = 0
         if (
-            current_platform.is_cuda_alike()
+            envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
+            and current_platform.is_cuda_alike()
             and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
         ):
             cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()
```

This alone is PR-worthy: today the pass runs (and poisons) even for users whose
estimate is discarded. It does not fix the poisoning for users who want the
estimate — that needs the real fix below — but it makes the blast radius opt-in
and gives affected users an env kill-switch
(`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`).

(Our fork additionally flips the env default to off; upstream will likely prefer
keeping it on plus the real fix.)

## Real fix (needs upstream design input)

`_teardown_profiling_state` must return the runner to a truly pre-KV-init state:

- reset lazily-initialized model-state contexts (`model_state`'s mamba
  spec-decode ctx and friends: `_mamba_ctx = None`, `is_initialized = False`,
  drop `state_base_addrs`/block-table pointer snapshots);
- tear down inductor cudagraph-tree recordings/static bindings created during
  the profiling capture (not just the `CUDAGraphWrapper`s);
- more robustly: introduce a `reset_for_reinit()` protocol on runner components
  so anything caching device pointers keyed to `initialize_kv_cache` must
  implement it, or run the profiling pass against a forked/isolated runner state
  so the real runner never observes throwaway tensors.

## Repro (our stack; should transfer to any hybrid GDN model)

```
vllm serve <Qwen3.5-hybrid-model> \
  --tensor-parallel-size 8 --pipeline-parallel-size 1 \
  --kv-cache-dtype int8_per_token_head \
  --max-model-len 1000000 <yarn overrides> \
  # default compilation config, i.e. piecewise graphs on
```
→ IMA at first warmup prefill, every boot.
Add `--compilation-config '{"cudagraph_mode": "NONE"}'` (or `-O0`, or
`--enforce-eager`) → boots and serves normally.
On base `62195e978` (pre-a4d70bef3) the default config boots and serves normally.

## Side findings from the same hunt (separate issues)

1. sm86-only builds of `299ebd094` fail to compile:
   `csrc/libtorch_stable/torch_bindings.cpp:812` references
   `fused_gdn_decode_post_conv_mtp` which is arch-gated out of the build for
   `TORCH_CUDA_ARCH_LIST=8.6` — binding registration lacks the matching guard.
   (Workaround: build with `8.6;9.0`.)
2. vllm-project/vllm#40756 (MTP draft input_ids stream race) is real and
   orthogonal; we carry the `torch.accelerator.current_stream().synchronize()`
   fence after draft buffer writes.
3. `mode 0 + cudagraph_mode FULL` additionally dies (silent SIGSEGV) after
   capture on this hybrid model even with the align ctx implicated—full-graph
   capture of hybrid GDN decode likely deserves its own guard/validation
   upstream (FULL_DECODE_ONLY same). May be the same dangling-pointer family
   (rows 8-9 ran before our fix); retest with the fix before reporting.

## Suggested upstream deliverables

1. PR: the minimal gating diff above (+ regression note in the docstring).
2. Issue (or follow-up PR): the teardown completeness problem, with this
   document's mechanism section; happy to test candidate fixes on our 8×A6000
   hybrid deployment, where the failure is 100% deterministic.
