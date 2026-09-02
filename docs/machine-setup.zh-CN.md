# 机器底座参考（解锁/驱动/Gen2/功耗/Docker/网络）

> 本文件是**裸机准备参考**：§0-3（解锁→Gen2→230W→Docker）适用于任何模型部署。
> §4 之后是纯文本模型(0731)的部署示例 — 与视觉服务**二选一**（显存互斥），仅作参考。

> 目标：在 4× 解锁 CMP 170HX 机器上部署 DeepSeek-V4-Flash-0731 生产推理服务。
> 命令均在 4×170HX / Ubuntu 26.04 / 内核 7.0 实测通过。按顺序执行，遇 [坑] 按提示处理。
> 部署前全文替换：`<user>`（SSH 用户）、`<IP>`、`<代理机>`（HTTP 代理 host:port，无则去掉相关行）。

## 0. 前置条件

- 4× CMP 170HX（`lspci -nn | grep 20c2`），Secure Boot 关闭（`mokutil --sb-state`）
- ≥3TB 空盘、≥200GB RAM、机器可达 SSH

```bash
# SSH 免密 + 免密 sudo
sshpass -p '<密码>' ssh <user>@<IP> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '<你的公钥>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
sshpass -p '<密码>' ssh <user>@<IP> "echo '<密码>' | sudo -S sh -c 'echo \"<user> ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/deploy-nopasswd && chmod 440 /etc/sudoers.d/deploy-nopasswd'"
```

## 1. 驱动 + 解锁

```bash
sudo apt-get install -y nvidia-driver-610-open linux-headers-$(uname -r) build-essential dkms git
# [坑] dpkg 卡 dracut "/boot no space"（/boot 仅 265MB）：删旧内核文件 + 禁 nvidia 进 initrd：
#   sudo rm -f /boot/*-旧版本* ; echo 'omit_drivers+=" nvidia nvidia_modeset nvidia_drm nvidia_uvm "' | sudo tee /etc/dracut.conf.d/omit-nvidia.conf
#   sudo dpkg --configure -a

mkdir -p ~/tools && cd ~/tools
git clone https://github.com/amoghmunikote/cmpunlocker.git
cd cmpunlocker && sudo ./install.sh        # 编译补丁模块 + IOMMU grub 配置, ~15min
# 冷重启（RTC 自动开机；256GB 内存 POST 训练 3-5 分钟，勿误判死机）
sudo rtcwake -m off -s 90
# 验证：
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 期望: 4 × "NVIDIA CMP 170HX, 65536 MiB"
sudo nvidia-smi -q | grep -c "HW Power Brake Slowdown  *: Not Active"   # 期望 4
```

## 2. PCIe Gen2 + 功耗帽（开机固化）

```bash
sudo tee /usr/local/sbin/gen2-ensure.sh > /dev/null <<'EOF'
#!/bin/bash
# 230W 帽（250W 帽瞬态会到 301W 触发 Xid43；230W 实测瞬态 ≤279W 安全）
# Gen2: 冷启动补丁的 PL 寄存器可能没写对，重载驱动后 hammer 第一轮即中
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HAMMER=/home/<user>/tools/cmpunlocker/tools/hammer.sh
LOG="gen2-ensure"
for g in 0 1 2 3; do nvidia-smi -i $g -pl 230 >/dev/null 2>&1; done
logger -t $LOG "power cap 230W applied"
gens=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits 2>/dev/null | sort -u | head -1)
if [ "$gens" != "2" ]; then
  logger -t $LOG "not Gen2 (=$gens), reloading nvidia + hammer"
  modprobe nvidia 2>/dev/null
  rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null
  modprobe nvidia nvidia_uvm 2>/dev/null; sleep 3
  "$HAMMER" >/dev/null 2>&1
fi
nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader | sort -u | head -1 | grep -q "^2$" \
  && logger -t $LOG "Gen2 ok" || logger -t $LOG "WARNING: still not Gen2"
exit 0
EOF
sudo chmod 700 /usr/local/sbin/gen2-ensure.sh
sudo tee /etc/systemd/system/gen2-ensure.service > /dev/null <<'EOF'
[Unit]
Description=CMP 170HX 230W cap + PCIe Gen2 ensure (before docker)
After=systemd-modules-load.service multi-user.target
Before=docker.service
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/gen2-ensure.sh
TimeoutStartSec=300
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now gen2-ensure.service
# 验证: nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader  → 全部 2
```

## 3. Docker（国内网络现实）

```bash
sudo apt-get install -y docker.io containerd
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
# 拉镜像必胜法：daemon 挂代理 + pull 套重试循环
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<'EOF'
[Service]
Environment="HTTP_PROXY=http://<代理机>"
Environment="HTTPS_PROXY=http://<代理机>"
Environment="NO_PROXY=localhost,127.0.0.1,.ustc.edu.cn,.gh-proxy.com"
EOF
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{ "runtimes": { "nvidia": { "args": [], "path": "nvidia-container-runtime" } } }
EOF
sudo systemctl daemon-reload && sudo systemctl restart docker
# [坑] auth.docker.io TLS 超时是间歇的：pull 一律套循环
#   for i in $(seq 1 15); do sudo docker pull <image> && break; sleep 8; done
# [坑] 别用 pkill -f "docker pull" —— 会匹配自己 shell 自杀会话（按 PID 杀）
```

## 4. 模型下载（~173GB）

```bash
python3 -m venv ~/hfenv && ~/hfenv/bin/pip install -U huggingface_hub
~/hfenv/bin/pip uninstall -y hf_xet hf_transfer    # [坑] Xet 协议与镜像不兼容 → 401
mkdir -p ~/models/dsv4-flash
HF_ENDPOINT=https://hf-mirror.com nohup ~/hfenv/bin/hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir ~/models/dsv4-flash > /tmp/dl.log 2>&1 &
# 期望 ~67MB/s；hf 自带校验，日志出现 "✓ Downloaded" 即完整
```

## 5. 引擎构建（c3046d1 + 补丁，sm80 全量编译 ~90min）

```bash
mkdir -p ~/tools && cd ~/tools
git clone https://github.com/allover326/deepseek-v4-cmp170hx.git

# 5.1 重建 c3046d1 基线（上游被 force-push，git 不可达，tarball 按树哈希校验）
git clone --branch dsv4-flash-a100 --single-branch https://github.com/haosdent/vllm.git
cd vllm
git fetch origin '+refs/*:refs/remotes/all/*'
git config user.email "deploy@local" && git config user.name "deploy"
curl -sL -o /tmp/c3046d1.tar.gz https://codeload.github.com/haosdent/vllm/tar.gz/c3046d1ebd2dae9b94ad2ef5f966ea153632251e
rm -rf /tmp/c3046d1-src && mkdir /tmp/c3046d1-src && tar xzf /tmp/c3046d1.tar.gz -C /tmp/c3046d1-src --strip-components=1
export GIT_INDEX_FILE=/tmp/c3046d1.index
git read-tree --empty && git --work-tree=/tmp/c3046d1-src add -Af
TREE=$(git write-tree); echo "$TREE"   # 必须等于 d13ae12b9a6621ef8d218f53741e59c6db2f68d2，不等则停
git tag c3046d1-recon "$(git commit-tree $TREE -p f8ea5bb163c161ef38b401d055cc5fd4a934091a -m recon)"
unset GIT_INDEX_FILE
git checkout -B rebase-c3046d1 c3046d1-recon

# 5.2 补丁 0002-0006（跳过 0001，已上游）
P=~/tools/deepseek-v4-cmp170hx/patches
for p in 0002-speculative 0003-pp_utils 0004-model_runner 0005-dspark-utils 0005a-prefill-topk-torch-fallback 0006-logits-row-chunk; do
  patch -p1 --forward < $P/$p.patch || { echo "PATCH FAILED: $p"; exit 1; }
done

# 5.3 构建文件（网络坑内置：rust 走 USTC 镜像、GitHub 克隆走 gh-proxy、pip 走代理）
cp ~/tools/deepseek-v4-cmp170hx/docker/Dockerfile.fullbuild .
cp ~/tools/deepseek-v4-cmp170hx/docker/dockerignore.txt .dockerignore
curl -sL -o rustup-init https://mirrors.ustc.edu.cn/rust-static/rustup/dist/x86_64-unknown-linux-gnu/rustup-init && chmod +x rustup-init
python3 - <<'PYEOF'
c = open("Dockerfile.fullbuild").read()
c = c.replace("ENV CUDA_HOME=/usr/local/cuda",
  "ENV CUDA_HOME=/usr/local/cuda\n"
  "ENV http_proxy=http://<代理机> https_proxy=http://<代理机> no_proxy=localhost,127.0.0.1,.ustc.edu.cn,.tuna.tsinghua.edu.cn,.gh-proxy.com")
old = "RUN curl --proto"
i = c.find(old); j = c.find("minimal", i) + len("minimal")
c = c[:i] + ("ENV RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static\n"
  "ENV RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup\n"
  "COPY rustup-init /tmp/rustup-init\n"
  "RUN chmod +x /tmp/rustup-init && /tmp/rustup-init -y --default-toolchain 1.95 --profile minimal") + c[j:]
marker = "RUN rustc --version && pip install --no-cache-dir -e . --no-build-isolation"
c = c.replace(marker,
  "RUN git config --global http.version HTTP/1.1 && git config --global http.postBuffer 524288000 && "
  "git config --global http.lowSpeedLimit 1000 && git config --global http.lowSpeedTime 60 && "
  "git config --global url.https://gh-proxy.com/https://github.com/.insteadOf https://github.com/\n" + marker)
open("Dockerfile.fullbuild","w").write(c)
print("dockerfile patched")
PYEOF
sudo setsid nohup docker build -f Dockerfile.fullbuild -t dsv4-a100:c3046d1 . > /tmp/build.log 2>&1 < /dev/null &
# 盯进度: tail -f /tmp/build.log；成功标志 = 最后打印 "custom ops import OK"
```

## 6. 启动（生产参数，全部必填）

```bash
cat > ~/tools/launch.sh <<'EOF'
#!/bin/bash
docker stop -t 60 dsv4 >/dev/null 2>&1; docker rm dsv4 >/dev/null 2>&1
docker run -d --name dsv4 --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e VLLM_MARLIN_FP8_DEQUANT_BF16=1 \
  -e VLLM_PP_LAYER_PARTITION=12,12,12,7 \
  -v /home/<user>/models/dsv4-flash:/model:ro \
  --shm-size=16g -p 8098:8000 \
  dsv4-a100:c3046d1 vllm serve /model --served-model-name dsv4s \
  --pipeline-parallel-size 4 --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 262144 --max-num-batched-tokens 2048 --trust-remote-code \
  --gpu-memory-utilization 0.92 --max-num-seqs 64 \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
EOF
chmod +x ~/tools/launch.sh && sudo ~/tools/launch.sh
```

**参数红线**（实测依据，违反即性能崩塌或炸卡）：
- PP=4，**禁止 TP**（Gen2x4 无 P2P：TP 解码 -60%，prefill 平躺）
- `VLLM_PP_LAYER_PARTITION=12,12,12,7` 必设（不设 KV 池 -35%）
- `DSV4_LOGITS_ROW_CHUNK=64` 必设（长会话 >718k CUDA 崩）
- 230W 帽必设（250W 帽瞬态 301W → Xid43）
- 禁 `--enforce-eager`（慢 12 倍）；禁 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`（GA100 CMP 直接报错）
- util 上限 0.92（0.95 有 cudagraph 捕获 OOM 风险）

## 7. 接入 litellm 网关

```yaml
# litellm config.yaml 的 model_list 下追加（api_base 换成本机实际地址）
- model_name: dsv4-flash
  litellm_params:
    model: openai/dsv4s
    api_base: http://<本机IP>:8098/v1
    api_key: none
    request_timeout: 3600
    stream_timeout: 3600
```
```
# [坑] 若目标 config.yaml 是"列表项零缩进"风格，插入必须零缩进（2 空格缩进会 YAML 解析失败）
# 改后重启 litellm，并跑一次 chat 验证路由
```

## 8. 验收清单

```bash
# 8.1 引擎就绪（启动到就绪 ~7min：载权重 + DSpark 抓图）
curl -s localhost:8098/health                                    # 200
docker logs dsv4 2>&1 | grep "Maximum concurrency"               # 期望 ~10.5x @262144
docker logs dsv4 2>&1 | grep "KV cache size"                     # 期望 ~2,754,423 tokens

# 8.2 冒烟
curl -s localhost:8098/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv4s","messages":[{"role":"user","content":"说ok"}],"max_tokens":20,"reasoning_effort":"none"}'

# 8.3 基准卫生守则：prompt 首 token 随机化（防前缀缓存命中出假快数字）；热机后测；token 按 usage 统计
```

**性能参考表（验收实测 @230W）**

| 场景 | 数值 |
|---|---|
| TTFT | 0.67s@1k / 3.4s@16k / 12.9s@64k / 47s@240k |
| 单流解码 | 113(代码) / 72(技术) / 52(中文) tok/s |
| 长文解码 | 14 @94k / 27 @229k tok/s |
| Agent 聚合 | 541@C8 / 1108@C32 / 1646@C64 tok/s，TTFT p95 <4s |
| 容量 | KV 池 275万 token；满 256k 会话 10.5 路 |
| 温度/功耗 | C=64 时 max 66°C；瞬态 ≤279W |

## 9. 运维

| 症状 | 处置 |
|---|---|
| Xid 43 / 引擎死 / CUDA 建上下文失败 | `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`，再启动；功耗帽被重置则重设 230W |
| prefill 莫名变慢（>30%） | KV 池碎片劣化，重启引擎即复；建议低峰周重启 |
| 驱动重载后 Gen2 丢失 | `sudo ~/tools/cmpunlocker/tools/hammer.sh`（重载后第一轮即中） |
| 长会话 >700k 崩 | 确认 `DSV4_LOGITS_ROW_CHUNK=64`；累积对话硬顶 ~1M |
| 精度敏感判断出错 | 请求带 `"chat_template_kwargs":{"thinking":true,"reasoning_effort":"high"}`（返回字段是 `reasoning` 非 `reasoning_content`） |

## 10. 已知上限（规划用）

- 深上下文（>100k）单流长生成 14-30 tok/s —— 所有开源引擎共性
- 满窗口并发 10.5 路物理上限（KV 密度 ~40KB/token 为开源现状）
- 3 卡可跑（`VLLM_PP_LAYER_PARTITION=15,15,13`），2 卡不行（权重 167GB）

---

## 11. 可探索调优（当前未应用）

### 硬件层面

| 调优 | 预期收益 | 前置/代价 |
|---|---|---|
| **PCIe 补焊电容**（B30 脚位 AC 耦合电容 ~24 颗 0402，配合 BIOS 强制 Gen4） | Gen2x4(2GB/s) → **Gen4x16(~16GB/s)**：模型加载 8×、prefill 大幅提升、TP 变可行 | 需焊接；参考 170th-Street wiki 的 pcie-capacitor-mod |
| **水冷改造**（Bykski N-TESLA-A100-X-V2 水头）或加强机箱风墙 | 温度余量 → 功耗帽可上探 250W+（当前最热卡持续 prefill 84°C 限制），prefill 再 +~10% | 改装成本；被动卡原设计靠服务器风道 |
| **涡轮风扇模组**（离心风扇直吹改装） | 同上，中低成本方案 | 噪音、占位 |
| **供电加固**（优质 EPS 线材/分线） | 瞬态余量更足，Xid 风险进一步下降 | 线材成本 |
| **170tune SM 降压**（`cachenetics/170tune`，逐卡 gate +150~+235） | 180W 帽场景 prefill +10% 级；230W 帽下降瞬态峰值 | 需逐卡资格测试（弱 SM 卡 +250 即 Xid13）；probe 期间必须停引擎 |
| **170tune HBM 超频**（NDIV 70） | 仅内存带宽受限场景；**解码 ~0 收益** | 需 FBPA 掩码（部分平台打不开）+ 8h 真实负载 gate，静默损坏风险 |
| Samsung 10GB 版卡 | 可解锁 80GB/卡（hynix 8G 上限 64GB） | 仅适用于该 SKU |

### 软件层面

| 调优 | 预期收益 | 前置/代价 |
|---|---|---|
| **等上游 KV 压缩存储**（vLLM/SGLang 落实 compress_ratios/滑窗到存储） | KV 密度 40KB→~10KB/token，**容量 ×4-7**（满窗口会话 10.5 → 40+ 路） | 纯等待；盯 haosdent fork / vllm-backport / sgl PR 动向 |
| **等上游稀疏注意力深度优化** | 长上下文解码 14-30 → 上探（深上下文真解） | 纯等待 |
| **LMCache CPU 外溢**（vllm-backport 引擎 + `--kv-offloading-backend lmcache --kv-offloading-size <GiB>`） | RAM 当二级 KV 池（+~200万 token 级），支撑 16×满256k 会话 | 换 B 引擎（解码 -30%）；建议双实例按上下文长度分流 |
| **prefill/decode 分离部署**（PD disagg） | 长 prefill 不再阻塞交互解码 | 双份权重显存或多机；SGLang DSpark disagg 路线较成熟 |
| **seqs 按负载画像切换**（8=延迟形态 / 64=吞吐形态） | 延迟形态单流 85-97 tok/s（吞吐形态 71-85） | 切换需重启 ~7min；负载画像明确时用 |
| **SGLang CP KV LayerSplit**（上下文并行） | PCIe 拓扑可能受益的第三种并行轴 | 上游目前在 Hopper；sm80 未验证 |
| **低峰定时重启引擎**（cron 周重启） | 规避 KV 碎片导致的 prefill 30% 劣化 | 一次 ~7min 停机窗口 |
| **双实例分工**（本机大模型 + 另一台小模型机负责低延迟交互） | 整体 fleet 延迟与吞吐同时最优 | 已是可行架构，按流量调度 |
