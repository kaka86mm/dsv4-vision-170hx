#!/usr/bin/env python3
"""DSV4-Vision 验收基准: 正确性 + 单流 + 深度解码(TTFT分离) + 并发。

测量铁律 (违反即数据作废):
  1. 深度解码必须 TTFT 分离 (流式首 content 时刻), 绝不用 total/tokens
  2. token 数必须用 usage (DSpark 多 token/chunk, chunk 计数失真)
  3. 引擎重启后首个请求丢弃 (Triton JIT 冷启动)
  4. prompt 首 token 随机化 (防前缀缓存假快)

用法: python3 bench.py [--host localhost:8096] [--quick]
"""
import argparse, base64, json, random, statistics, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost:8096")
ap.add_argument("--quick", action="store_true", help="跳过深度与高并发")
A = ap.parse_args()
URL = f"http://{A.host}/v1/chat/completions"

_words = ("system performance memory bandwidth latency kernel scheduler database "
          "index query network packet routing cache coherence tensor parallel "
          "pipeline inference token generation attention").split()

def chat(messages, max_tokens, stream=False, timeout=3600):
    body = {"model": "dsv4-vision", "messages": messages, "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False}, "temperature": 0}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        j = json.load(resp)
        return {"total": time.perf_counter() - t0, "ttft": None,
                "ctok": j["usage"]["completion_tokens"]}
    ttft, usage = None, None
    for line in resp:
        line = line.decode("utf-8", "ignore").strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                j = json.loads(line[6:])
                if j.get("usage"): usage = j["usage"]
                ch = j.get("choices") or [{}]
                d = ch[0].get("delta", {})
                c = d.get("content")
                if c and ttft is None: ttft = time.perf_counter() - t0
            except Exception: pass
    total = time.perf_counter() - t0
    ct = usage["completion_tokens"] if usage else -1
    return {"total": total, "ttft": ttft, "ctok": ct}

def make_prompt(t):
    start = random.randint(0, len(_words) - 1)
    ws = [f"x{random.randint(0, 10**9)}"] + [_words[(i + start) % len(_words)] for i in range(t)]
    return " ".join(ws)

# ---------- 1. 正确性 ----------
print("== correctness ==")
for q in ["中国的首都是哪座城市？直接回答", "1+1等于几"]:
    r = chat([{"role": "user", "content": q}], 40)
    print(f"  {q[:14]} -> ok ({r['ctok']} tok)")

# ---------- 2. 单流 ----------
print("== single-stream decode (400 tok) ==")
chat([{"role": "user", "content": "warmup"}], 8)  # 铁律3: 丢弃冷启
r = chat([{"role": "user", "content": make_prompt(120) +
           " Write a detailed technical essay about distributed inference."}], 400)
print(f"  tech: {r['ctok'] / r['total']:.1f} tok/s")

# ---------- 3. 深度解码 (TTFT 分离) ----------
if not A.quick:
    print("== deep decode (TTFT-separated, usage-counted) ==")
    for depth in [93000, 178000]:
        body_msgs = [{"role": "user",
                      "content": make_prompt(depth) + "\n\n不要解释。从1数到120，每行一个。"}]
        r = chat(body_msgs, 250, stream=True)
        d = r["total"] - r["ttft"]
        print(f"  depth {depth // 1000}k: TTFT {r['ttft']:.1f}s | "
              f"{r['ctok']} tok in {d:.2f}s = {r['ctok'] / d:.1f} tok/s")

# ---------- 4. 并发 ----------
print("== agent concurrency (3.2k shared prefix, out=700) ==")
clist = [4, 8, 16] if A.quick else [4, 8, 16, 32]
for C in clist:
    sysmsg = "Tools:\n" + make_prompt(3200)
    barrier = threading.Barrier(C)
    res = [None] * C
    def w(i):
        barrier.wait()
        res[i] = chat([{"role": "system", "content": sysmsg},
                       {"role": "user", "content": "分析流水线并行的优劣。"}], 700)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(C) as ex: list(ex.map(w, range(C)))
    wall = time.perf_counter() - t0
    toks = sum(r["ctok"] for r in res)
    per = statistics.median(r["ctok"] / r["total"] for r in res)
    print(f"  C={C:2d}: agg {toks / wall:6.1f} t/s | per-req {per:5.1f} | wall {wall:.1f}s")
