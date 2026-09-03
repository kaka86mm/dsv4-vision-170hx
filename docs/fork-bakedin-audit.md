# Fork baked-in audit (snapshot root vs upstream base 62195e978)

Full delta = 124 files. Rerun `python3` coverage check after any rebase.

- 110 files: >=90% of fork-added lines present verbatim on master.
- Justified exceptions (feature present in different form, or superseded):
  - `.github/workflows.disabled/{notify-ci-authorized,pre-commit,record-ci-approval}.yml` — upstream's newer copies, parked disabled (policy: only docker-publish active).
  - `vllm/v1/worker/gpu/pp_utils.py` — fork's pre-PR46994 draft-token broadcast; superseded by PR46994's broadcast_draft (ported).
  - `vllm/v1/worker/utils.py` — fork's chunk-based zeroer kernel superseded by upstream's segment kernel; the fork's GROUP-SCOPED dispatch (#50576) re-implemented on top (`_group_meta`, `zero_block_ids(list[list[int]])`, `build_meta`).
  - `vllm/v1/worker/gpu/sample/gumbel.py` + `rejection_sampler_utils.py` — #50843 argmax clamps ported; surrounding old-structure lines superseded.
  - `vllm/v1/worker/gpu/model_runner.py` — pre-PR46994 scatter plumbing dropped for PR46994 form; eagle3-aux-layer PP guard ported.
  - `vllm/models/deepseek_v4/attention.py`, `amd/rocm.py`, `v1/attention/ops/rocm_aiter_mla_sparse.py`, `tokenizers/deepseek_v4_encoding.py` — upstream absorbed the fork content and evolved past it (eager-break capture regions, gfx950 paths, raise-instead-of-assert); ours kept.
  - test files (`test_kv_block_zeroer`, `test_compressor_kv_cache`, `test_fused_indexer_q_rope_quant`) — merged onto upstream-evolved internals; fork-only tests (cross-group guard, helpers) kept.
