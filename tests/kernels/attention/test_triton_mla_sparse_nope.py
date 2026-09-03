import torch, sys
sys.path.insert(0, "/home/mio/vllm-backport")
from vllm.v1.attention.ops.triton_mla_sparse_kernel import triton_mla_sparse_attention

torch.manual_seed(0)
dev = "cuda"

def ref(q, kv, idx, sm_scale, dv=512):
    # q [T,H,D], kv [N,1,D], idx [T,1,K]
    T, H, D = q.shape
    K = idx.shape[-1]
    out = torch.zeros(T, H, dv, dtype=torch.float32, device=q.device)
    for t in range(T):
        sel = idx[t, 0]
        valid = (sel >= 0) & (sel < kv.shape[0])
        s = sel[valid].long()
        if s.numel() == 0:
            continue
        k = kv[s, 0].float()                    # [k, D]
        qk = (q[t].float() @ k.T) * sm_scale    # [H, k]
        p = torch.softmax(qk, dim=-1)
        out[t] = p @ k[:, :dv]
    return out

for dim_qk, name in ((576, "RoPE 576 (regression)"), (512, "NoPE 512 (GLM-5.3-Flash)")):
    T, H, N, K = 4, 16, 2048, 128
    q = torch.randn(T, H, dim_qk, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(N, 1, dim_qk, dtype=torch.bfloat16, device=dev)
    idx = torch.stack([torch.randperm(N, device=dev)[:K].int() for _ in range(T)])
    idx = idx.view(T, 1, K)
    sm = dim_qk ** -0.5
    for splits in (1, 4):
        o = triton_mla_sparse_attention(q, kv, idx, sm_scale=sm, num_kv_splits=splits)
        r = ref(q, kv, idx, sm)
        err = (o.float() - r).abs().max().item()
        rel = err / r.abs().max().item()
        status = "PASS" if rel < 2e-2 else "FAIL"
        print(f"{status}  {name:32s} splits={splits}  out={tuple(o.shape)} max_abs_err={err:.4f} rel={rel:.2e}")
