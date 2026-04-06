#!/usr/bin/env python3
"""Generate days 3-100 notebooks for 100-days-of-inference."""
import json, os, sys

BASE = "/home/nvidia/src/github.com/elizabetht/100-days-of-inference"

def cell_md(src, cid):
    return {"cell_type":"markdown","metadata":{},"source": src if isinstance(src,list) else [src],"id":cid}

def cell_code(src, cid):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src if isinstance(src,list) else [src],"id":cid}

def make_nb(cells):
    return {
        "nbformat":4,"nbformat_minor":5,
        "metadata":{
            "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python","version":"3.10.0"}
        },
        "cells": cells
    }

def write_nb(day, slug, cells):
    d = f"{BASE}/day{day:02d}"
    os.makedirs(d, exist_ok=True)
    path = f"{d}/{slug}.ipynb"
    nb = make_nb(cells)
    with open(path,"w") as f:
        json.dump(nb, f, indent=1)
    return path

def write_readme(day, topic, layer, summary, insight, slug, key_demo, refs=""):
    d = f"{BASE}/day{day:02d}"
    os.makedirs(d, exist_ok=True)
    ref_section = refs if refs else f"- *Inference Engineering* (Philip Kiely, Baseten Books 2026)"
    content = f"""# Day {day:02d}

**Topic:** {topic}
**Date:** 2026-04-06
**Layer:** {layer}

## What I explored
{summary}

## Key insight
{insight}

## Code / experiment
Notebook: [`{slug}.ipynb`](./{slug}.ipynb)
Key demo: {key_demo}

## References
{ref_section}
"""
    with open(f"{d}/README.md","w") as f:
        f.write(content)

def validate(path):
    try:
        with open(path) as f:
            json.load(f)
        return True
    except Exception as e:
        print(f"INVALID {path}: {e}")
        return False

################################################################################
# DAY 03: MoE Routing
################################################################################
def day03():
    cells = [
        cell_md([
            "# Day 03: Mixture of Experts (MoE) Routing\n",
            "> *Inference Engineering* — Chapter 2.2.4 | Philip Kiely, Baseten Books 2026\n",
            "\n**Layer:** Runtime | **Prerequisite:** Day 02 (Transformer Blocks)\n"
        ], "c01"),
        cell_md([
            "## Concept Overview\n",
            "\n",
            "Mixture of Experts (MoE) replaces the dense FFN layer in a transformer with N separate expert FFNs, ",
            "routing each token to only K of them (typically K=2). This allows models to scale parameter count ",
            "without proportionally scaling compute — a Mixtral 8x7B has 47B parameters but uses ~13B per forward pass. ",
            "The routing is learned: a gating network produces logits over experts, and top-K selection determines ",
            "which experts process each token. Load balancing losses prevent router collapse (all tokens going to one expert).\n"
        ], "c02"),
        cell_code([
            "!pip install -q torch numpy matplotlib\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "print(f'CUDA available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
        ], "c03"),
        cell_md([
            "## 1. Top-K Gating\n",
            "\n",
            "The router is a linear layer mapping hidden states to expert logits. ",
            "Softmax + top-K selects which experts process each token, and their outputs are weighted-summed.\n",
            "\n",
            "$$\\text{gate}(x) = \\text{TopK}(\\text{softmax}(W_g x), K)$$\n"
        ], "c04"),
        cell_code([
            "class TopKRouter(nn.Module):\n",
            "    def __init__(self, d_model, num_experts, top_k):\n",
            "        super().__init__()\n",
            "        self.gate = nn.Linear(d_model, num_experts, bias=False)\n",
            "        self.top_k = top_k\n",
            "        self.num_experts = num_experts\n",
            "\n",
            "    def forward(self, x):\n",
            "        # x: [batch*seq, d_model]\n",
            "        logits = self.gate(x)  # [B*T, num_experts]\n",
            "        scores = F.softmax(logits, dim=-1)\n",
            "        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)\n",
            "        # Normalize top-k weights\n",
            "        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)\n",
            "        return topk_weights, topk_idx\n",
            "\n",
            "router = TopKRouter(d_model=256, num_experts=8, top_k=2)\n",
            "x = torch.randn(32, 256)  # 32 tokens\n",
            "weights, indices = router(x)\n",
            "print(f'Router input shape: {x.shape}')\n",
            "print(f'Top-2 weights shape: {weights.shape}')\n",
            "print(f'Top-2 indices shape: {indices.shape}')\n",
            "print(f'Sample token expert assignments: {indices[:4]}')\n",
            "print(f'Expert load distribution: {torch.bincount(indices.flatten(), minlength=8)}')\n"
        ], "c05"),
        cell_md([
            "## 2. Expert FFN and MoE Layer\n",
            "\n",
            "Each expert is a standard FFN. The MoE layer dispatches tokens to their assigned experts, ",
            "runs them in parallel, and combines the results with the gating weights.\n"
        ], "c06"),
        cell_code([
            "class ExpertFFN(nn.Module):\n",
            "    def __init__(self, d_model, d_ff):\n",
            "        super().__init__()\n",
            "        self.fc1 = nn.Linear(d_model, d_ff)\n",
            "        self.fc2 = nn.Linear(d_ff, d_model)\n",
            "\n",
            "    def forward(self, x):\n",
            "        return self.fc2(F.gelu(self.fc1(x)))\n",
            "\n",
            "class MoELayer(nn.Module):\n",
            "    def __init__(self, d_model, d_ff, num_experts, top_k):\n",
            "        super().__init__()\n",
            "        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])\n",
            "        self.router = TopKRouter(d_model, num_experts, top_k)\n",
            "        self.num_experts = num_experts\n",
            "        self.top_k = top_k\n",
            "\n",
            "    def forward(self, x):\n",
            "        B, T, D = x.shape\n",
            "        x_flat = x.view(B*T, D)\n",
            "        weights, indices = self.router(x_flat)  # [B*T, k], [B*T, k]\n",
            "        out = torch.zeros_like(x_flat)\n",
            "        for k in range(self.top_k):\n",
            "            for e in range(self.num_experts):\n",
            "                mask = (indices[:, k] == e)\n",
            "                if mask.any():\n",
            "                    expert_out = self.experts[e](x_flat[mask])\n",
            "                    out[mask] += weights[mask, k:k+1] * expert_out\n",
            "        return out.view(B, T, D)\n",
            "\n",
            "moe = MoELayer(d_model=256, d_ff=1024, num_experts=8, top_k=2)\n",
            "x = torch.randn(2, 16, 256)  # batch=2, seq=16, d=256\n",
            "out = moe(x)\n",
            "print(f'MoE input:  {x.shape}')\n",
            "print(f'MoE output: {out.shape}')\n",
            "params_dense = 2 * 256 * 1024  # one FFN\n",
            "params_moe = 8 * 2 * 256 * 1024  # 8 experts\n",
            "print(f'Dense FFN params: {params_dense:,}')\n",
            "print(f'MoE params (8 experts): {params_moe:,}')\n",
            "print(f'Active params per token: {params_dense * 2:,} (top-2 experts)')\n",
            "print(f'Parameter efficiency ratio: {params_moe / (params_dense * 2):.1f}x more params, same compute')\n"
        ], "c07"),
        cell_md([
            "## 3. Load Balancing Loss\n",
            "\n",
            "Without regularization, the router collapses — popular experts get stronger gradients and become more popular. ",
            "The auxiliary load balancing loss penalizes uneven expert utilization:\n",
            "\n",
            "$$\\mathcal{L}_{\\text{aux}} = \\alpha \\cdot N \\sum_{i=1}^{N} f_i \\cdot P_i$$\n",
            "\n",
            "where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the mean routing probability.\n"
        ], "c08"),
        cell_code([
            "def load_balance_loss(router_logits, num_experts, top_k):\n",
            "    \"\"\"Compute auxiliary load balancing loss (Switch Transformer style).\"\"\"\n",
            "    scores = F.softmax(router_logits, dim=-1)  # [N, E]\n",
            "    _, indices = torch.topk(scores, top_k, dim=-1)\n",
            "    # f_i: fraction of tokens routed to expert i\n",
            "    one_hot = F.one_hot(indices, num_experts).float()  # [N, k, E]\n",
            "    f = one_hot.sum(dim=1).mean(dim=0)  # [E]\n",
            "    # P_i: mean routing probability for expert i\n",
            "    P = scores.mean(dim=0)  # [E]\n",
            "    loss = num_experts * (f * P).sum()\n",
            "    return loss\n",
            "\n",
            "# Simulate balanced vs collapsed routing\n",
            "N = 1000  # tokens\n",
            "E = 8     # experts\n",
            "\n",
            "# Balanced: roughly uniform\n",
            "balanced_logits = torch.randn(N, E)\n",
            "loss_balanced = load_balance_loss(balanced_logits, E, top_k=2)\n",
            "\n",
            "# Collapsed: all mass on expert 0\n",
            "collapsed_logits = torch.randn(N, E)\n",
            "collapsed_logits[:, 0] += 10.0\n",
            "loss_collapsed = load_balance_loss(collapsed_logits, E, top_k=2)\n",
            "\n",
            "print(f'Load balance loss (balanced routing):  {loss_balanced:.4f}')\n",
            "print(f'Load balance loss (collapsed routing): {loss_collapsed:.4f}')\n",
            "print(f'Ratio: {loss_collapsed/loss_balanced:.1f}x higher for collapsed')\n",
            "\n",
            "# Visualize expert load distributions\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "for ax, logits, title in zip(axes, [balanced_logits, collapsed_logits], ['Balanced', 'Collapsed']):\n",
            "    scores = F.softmax(logits, dim=-1)\n",
            "    _, idx = torch.topk(scores, 2, dim=-1)\n",
            "    counts = torch.bincount(idx.flatten(), minlength=E).float()\n",
            "    ax.bar(range(E), counts / counts.sum())\n",
            "    ax.set_xlabel('Expert ID'); ax.set_ylabel('Fraction of tokens')\n",
            "    ax.set_title(f'{title} routing')\n",
            "    ax.axhline(1/E, color='red', linestyle='--', label='Uniform')\n",
            "    ax.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig('moe_routing_balance.png', dpi=100, bbox_inches='tight')\n",
            "plt.show()\n",
            "print('Saved moe_routing_balance.png')\n"
        ], "c09"),
        cell_md([
            "## 4. MoE vs Dense: Parameter Efficiency\n",
            "\n",
            "Real-world MoE models like Mixtral 8x7B and DeepSeek MoE demonstrate that sparse activation ",
            "achieves quality comparable to dense models with 2-4x fewer active FLOPs.\n"
        ], "c10"),
        cell_code([
            "# Compare parameter counts and active compute for real model configs\n",
            "configs = {\n",
            "    'Llama-3-8B (dense)': {'layers': 32, 'd_model': 4096, 'd_ff': 14336, 'num_experts': 1, 'top_k': 1},\n",
            "    'Mixtral-8x7B (MoE)': {'layers': 32, 'd_model': 4096, 'd_ff': 14336, 'num_experts': 8, 'top_k': 2},\n",
            "    'DeepSeek-MoE-16B':   {'layers': 28, 'd_model': 2048, 'd_ff': 1408,  'num_experts': 64, 'top_k': 6},\n",
            "}\n",
            "\n",
            "print(f'{\"Model\":<30} {\"Total FFN Params\":>18} {\"Active FFN Params\":>20} {\"Efficiency\":>12}')\n",
            "print('-' * 82)\n",
            "for name, cfg in configs.items():\n",
            "    total = cfg['layers'] * cfg['num_experts'] * 2 * cfg['d_model'] * cfg['d_ff']\n",
            "    active = cfg['layers'] * cfg['top_k'] * 2 * cfg['d_model'] * cfg['d_ff']\n",
            "    eff = total / active\n",
            "    print(f'{name:<30} {total/1e9:>17.2f}B {active/1e9:>19.2f}B {eff:>11.1f}x')\n"
        ], "c11"),
        cell_md([
            "## Experiments: Try These\n",
            "\n",
            "1. **Vary top-K**: Change `top_k` from 1 to 4 and measure how load distribution changes with the same router.\n",
            "2. **Expert specialization**: Train a tiny MoE on synthetic data (e.g., odd vs even numbers) and inspect which expert handles which input type.\n",
            "3. **Router noise**: Add Gaussian noise to router logits during training (as in Switch Transformer) and observe its effect on load balancing.\n"
        ], "c12"),
        cell_md([
            "## Key Takeaways\n",
            "\n",
            "- MoE replaces the dense FFN with N experts and routes each token to only K of them, enabling parameter scaling without proportional compute scaling.\n",
            "- The router is a learned linear layer; top-K selection with weight normalization controls expert mixing.\n",
            "- Without load balancing loss, routers collapse to a few popular experts — the auxiliary loss enforces uniform utilization.\n",
            "- Real MoE models (Mixtral 8x7B, DeepSeek) achieve dense-model quality at 2-4x lower active FLOP counts.\n",
            "\n",
            "## References\n",
            "- *Inference Engineering* Ch 2.2.4 — Philip Kiely, Baseten Books 2026\n",
            "- Shazeer et al. (2017), \"Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer\"\n",
            "- Fedus et al. (2021), \"Switch Transformers\"\n",
            "- Jiang et al. (2024), \"Mixtral of Experts\"\n"
        ], "c13"),
    ]
    p = write_nb(3, "moe-routing", cells)
    write_readme(3, "Mixture of Experts (MoE) Routing", "Runtime",
        "Studied how MoE layers replace dense FFNs with N sparse experts, routing each token to only top-K of them via a learned gating network. Implemented a complete MoE layer with top-K routing, load balancing loss, and parameter efficiency analysis comparing real model configs (Llama-3-8B vs Mixtral 8x7B).",
        "MoE achieves 4x more parameters than a dense model while using the same FLOPs per token — quality scales with total parameters, but inference cost only scales with active parameters.",
        "moe-routing",
        "Expert load distribution visualization under balanced vs collapsed routing",
        "- *Inference Engineering* Ch 2.2.4 (Philip Kiely, Baseten Books 2026)\n- Shazeer et al. (2017), \"Outrageously Large Neural Networks\"\n- Fedus et al. (2021), \"Switch Transformers\"")
    return validate(p)

################################################################################
# DAY 04: Ops:Byte Ratio & Arithmetic Intensity
################################################################################
def day04():
    cells = [
        cell_md([
            "# Day 04: Ops:Byte Ratio & Arithmetic Intensity\n",
            "> *Inference Engineering* — Chapter 2.4 | Philip Kiely, Baseten Books 2026\n",
            "\n**Layer:** Runtime | **Prerequisite:** Day 03 (MoE)\n"
        ], "c01"),
        cell_md([
            "## Concept Overview\n",
            "\n",
            "Arithmetic intensity (ops:byte ratio) is the number of floating point operations performed per byte of memory transferred. ",
            "It determines whether a kernel is compute-bound or memory-bound. ",
            "The roofline model plots achievable FLOP/s as a function of arithmetic intensity, with two ceilings: the memory bandwidth ceiling and the compute ceiling. ",
            "For LLM inference, the decode phase (batch=1) is deeply memory-bound — loading model weights dominates. ",
            "Batching increases arithmetic intensity by amortizing weight loads across multiple tokens.\n"
        ], "c02"),
        cell_code([
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "\n",
            "print(f'CUDA available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
            "    props = torch.cuda.get_device_properties(0)\n",
            "    print(f'Memory: {props.total_memory/1e9:.1f} GB')\n"
        ], "c03"),
        cell_md([
            "## 1. The Roofline Model\n",
            "\n",
            "For a kernel with arithmetic intensity $I$ (FLOP/byte):\n",
            "\n",
            "$$\\text{Attainable FLOP/s} = \\min(\\text{Peak FLOP/s},\\ I \\times \\text{Memory BW})$$\n",
            "\n",
            "The ridge point $I^* = \\text{Peak FLOP/s} / \\text{Memory BW}$ separates memory-bound (left) from compute-bound (right).\n"
        ], "c04"),
        cell_code([
            "# GPU specs: A100 80GB SXM\n",
            "gpu_specs = {\n",
            "    'A100 80GB': {'tflops_fp16': 312, 'mem_bw_tbs': 2.0},   # 2 TB/s HBM2e\n",
            "    'H100 SXM':  {'tflops_fp16': 989, 'mem_bw_tbs': 3.35},  # 3.35 TB/s HBM3\n",
            "    'DGX Spark': {'tflops_fp16': 67,  'mem_bw_tbs': 0.273}, # GB10 ~273 GB/s\n",
            "}\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "intensities = np.logspace(-1, 3, 500)  # 0.1 to 1000 FLOP/byte\n",
            "\n",
            "colors = ['blue', 'orange', 'green']\n",
            "for (name, specs), color in zip(gpu_specs.items(), colors):\n",
            "    peak = specs['tflops_fp16'] * 1e12  # FLOP/s\n",
            "    bw = specs['mem_bw_tbs'] * 1e12    # byte/s\n",
            "    ridge = peak / bw\n",
            "    attainable = np.minimum(peak, intensities * bw)\n",
            "    ax.loglog(intensities, attainable / 1e12, label=f'{name} (ridge={ridge:.0f} FLOP/B)', color=color)\n",
            "    ax.axvline(ridge, color=color, linestyle=':', alpha=0.5)\n",
            "\n",
            "# Mark typical LLM operations\n",
            "ops = {\n",
            "    'Decode (bs=1)': 1,\n",
            "    'Decode (bs=32)': 32,\n",
            "    'Prefill (seq=512)': 512,\n",
            "    'GEMM (large)': 1000,\n",
            "}\n",
            "for label, intensity in ops.items():\n",
            "    ax.axvline(intensity, color='red', linestyle='--', alpha=0.3)\n",
            "    ax.text(intensity * 1.1, 1, label, rotation=90, fontsize=8, color='red')\n",
            "\n",
            "ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')\n",
            "ax.set_ylabel('Attainable FLOP/s (TFLOP/s)')\n",
            "ax.set_title('Roofline Model — LLM Inference Regimes')\n",
            "ax.legend()\n",
            "ax.grid(True, which='both', alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.savefig('roofline_model.png', dpi=100, bbox_inches='tight')\n",
            "plt.show()\n",
            "print('Saved roofline_model.png')\n"
        ], "c05"),
        cell_md([
            "## 2. Arithmetic Intensity of LLM Operations\n",
            "\n",
            "A matrix multiply $y = Wx$ where $W \\in \\mathbb{R}^{m \\times k}$:\n",
            "- **FLOPs:** $2mk$ (multiply-accumulate)\n",
            "- **Bytes read:** $(mk + k) \\times \\text{dtype\\_size}$\n",
            "- **Arithmetic intensity:** $\\approx 2mk / (mk \\cdot \\text{dtype\\_size}) = 2/\\text{dtype\\_size}$ for decode (batch=1)\n",
            "\n",
            "For batch size $B$: intensity $\\approx 2B/\\text{dtype\\_size}$. This is why batching is the primary lever for moving from memory-bound to compute-bound.\n"
        ], "c06"),
        cell_code([
            "def matmul_intensity(m, k, batch, dtype_bytes=2):\n",
            "    \"\"\"Arithmetic intensity for batched matmul y=xW, x:[batch,k], W:[k,m]\"\"\"\n",
            "    flops = 2 * batch * m * k\n",
            "    # Weight W dominates reads when batch is small\n",
            "    bytes_read = (m * k + batch * k + batch * m) * dtype_bytes\n",
            "    return flops / bytes_read\n",
            "\n",
            "# Llama-3-8B layer: d_model=4096, d_ff=14336\n",
            "d_model, d_ff = 4096, 14336\n",
            "print('Arithmetic Intensity for Llama-3-8B FFN layer (W:[d_ff, d_model])')\n",
            "print(f'{\"Batch Size\":>12} {\"Intensity (FLOP/B)\":>20} {\"Regime\":>15}')\n",
            "print('-' * 50)\n",
            "ridge_a100 = 312e12 / 2e12  # 156 FLOP/byte for A100\n",
            "for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:\n",
            "    intensity = matmul_intensity(d_ff, d_model, bs)\n",
            "    regime = 'COMPUTE-BOUND' if intensity > ridge_a100 else 'memory-bound'\n",
            "    print(f'{bs:>12} {intensity:>20.1f} {regime:>15}')\n"
        ], "c07"),
        cell_md([
            "## 3. Measuring Real Memory Bandwidth\n",
            "\n",
            "We can empirically measure how close to peak bandwidth a simple memory-bound operation achieves.\n"
        ], "c08"),
        cell_code([
            "import time\n",
            "\n",
            "def measure_bandwidth(size_mb=1024, dtype=torch.float16, device='cpu'):\n",
            "    \"\"\"Measure achievable memory bandwidth via a vector copy.\"\"\"\n",
            "    n = size_mb * 1024 * 1024 // 2  # FP16 = 2 bytes\n",
            "    a = torch.randn(n, dtype=dtype, device=device)\n",
            "    b = torch.empty_like(a)\n",
            "\n",
            "    # Warmup\n",
            "    for _ in range(3):\n",
            "        b.copy_(a)\n",
            "    if device != 'cpu':\n",
            "        torch.cuda.synchronize()\n",
            "\n",
            "    t0 = time.perf_counter()\n",
            "    iters = 10\n",
            "    for _ in range(iters):\n",
            "        b.copy_(a)\n",
            "    if device != 'cpu':\n",
            "        torch.cuda.synchronize()\n",
            "    t1 = time.perf_counter()\n",
            "\n",
            "    bytes_moved = 2 * n * 2 * iters  # read + write, FP16\n",
            "    bw_gb_s = bytes_moved / (t1 - t0) / 1e9\n",
            "    return bw_gb_s\n",
            "\n",
            "cpu_bw = measure_bandwidth(256, device='cpu')\n",
            "print(f'CPU memory bandwidth: {cpu_bw:.1f} GB/s')\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    gpu_bw = measure_bandwidth(1024, device='cuda')\n",
            "    print(f'GPU memory bandwidth: {gpu_bw:.1f} GB/s')\n",
            "    props = torch.cuda.get_device_properties(0)\n",
            "    print(f'GPU peak BW (spec): ~273 GB/s for DGX Spark GB10')\n",
            "    print(f'Efficiency: {gpu_bw/273*100:.0f}% of peak')\n"
        ], "c09"),
        cell_md([
            "## Experiments: Try These\n",
            "\n",
            "1. **Measure GEMM efficiency**: Use `torch.utils.benchmark` to measure FLOP/s for matrix multiplies at different sizes. Compare against theoretical peak.\n",
            "2. **Batch size sweep**: Run a decode-style matmul (tall, skinny W) across batch sizes 1-256 and plot latency vs throughput — find the knee of the curve.\n",
            "3. **Mixed precision**: Compare arithmetic intensity for FP32 vs FP16 vs INT8. How does quantization affect the memory-bound threshold?\n"
        ], "c10"),
        cell_md([
            "## Key Takeaways\n",
            "\n",
            "- Arithmetic intensity = FLOPs / bytes transferred; it determines whether a kernel is compute-bound or memory-bound.\n",
            "- The roofline model shows attainable FLOP/s = min(peak compute, intensity × memory BW).\n",
            "- LLM decode with batch=1 has intensity ~1 FLOP/byte — deeply memory-bound; the GPU spends most time waiting on HBM.\n",
            "- Increasing batch size linearly increases arithmetic intensity — the primary mechanism for improving GPU utilization during inference.\n",
            "\n",
            "## References\n",
            "- *Inference Engineering* Ch 2.4 — Philip Kiely, Baseten Books 2026\n",
            "- Williams et al. (2009), \"Roofline: An Insightful Visual Performance Model\"\n",
            "- NVIDIA A100 Datasheet, NVIDIA H100 Datasheet\n"
        ], "c11"),
    ]
    p = write_nb(4, "ops-byte-ratio", cells)
    write_readme(4, "Ops:Byte Ratio & Arithmetic Intensity", "Runtime",
        "Explored the roofline model and arithmetic intensity as tools for understanding whether LLM inference is compute-bound or memory-bound. Built an interactive roofline plot for A100/H100/DGX Spark GPUs and computed arithmetic intensity across batch sizes for a Llama-3-8B FFN layer.",
        "LLM decode at batch=1 has ~1 FLOP/byte arithmetic intensity — 100x below the A100 ridge point — making it purely memory-bound. Batching is the primary lever to improve GPU utilization.",
        "ops-byte-ratio",
        "Roofline model plot with real GPU specs and LLM operation intensity markers",
        "- *Inference Engineering* Ch 2.4 (Philip Kiely, Baseten Books 2026)\n- Williams et al. (2009), \"Roofline: An Insightful Visual Performance Model\"")
    return validate(p)

################################################################################
# DAY 05: CUDA Kernels, Kernel Selection & Fusion
################################################################################
def day05():
    cells = [
        cell_md([
            "# Day 05: CUDA Kernels, Kernel Selection & Kernel Fusion\n",
            "> *Inference Engineering* — Chapter 4.1 | Philip Kiely, Baseten Books 2026\n",
            "\n**Layer:** Runtime | **Prerequisite:** Day 04 (Arithmetic Intensity)\n"
        ], "c01"),
        cell_md([
            "## Concept Overview\n",
            "\n",
            "A CUDA kernel is a function that runs in parallel across thousands of GPU threads. ",
            "PyTorch dispatches operations to kernels via its dispatcher based on device, dtype, and layout. ",
            "Kernel fusion combines multiple operations into a single kernel pass, reducing memory round-trips. ",
            "For memory-bound operations (like elementwise activations), fusion is transformative: instead of reading and writing tensors N times, ",
            "you read once, compute all ops in registers, and write once. FlashAttention is the canonical example of kernel fusion in LLM inference.\n"
        ], "c02"),
        cell_code([
            "import torch\n",
            "import torch.nn.functional as F\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import time\n",
            "\n",
            "print(f'CUDA available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
            "    print(f'PyTorch version: {torch.__version__}')\n"
        ], "c03"),
        cell_md([
            "## 1. Kernel Dispatch Overhead\n",
            "\n",
            "Each PyTorch operation incurs kernel launch overhead (~5-20 µs). For small tensors, this overhead dominates. ",
            "Measuring dispatch cost reveals where fusion matters most.\n"
        ], "c04"),
        cell_code([
            "def benchmark_op(fn, *args, warmup=10, iters=100, device='cpu'):\n",
            "    for _ in range(warmup):\n",
            "        fn(*args)\n",
            "    if device == 'cuda':\n",
            "        torch.cuda.synchronize()\n",
            "    t0 = time.perf_counter()\n",
            "    for _ in range(iters):\n",
            "        fn(*args)\n",
            "    if device == 'cuda':\n",
            "        torch.cuda.synchronize()\n",
            "    return (time.perf_counter() - t0) / iters * 1e6  # microseconds\n",
            "\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "sizes = [64, 256, 1024, 4096, 16384]\n",
            "results = {'unfused': [], 'fused_approx': []}\n",
            "\n",
            "for n in sizes:\n",
            "    x = torch.randn(n, device=device)\n",
            "    # Unfused: separate ops, each hitting HBM\n",
            "    t_unfused = benchmark_op(lambda a: F.gelu(a * 2.0 + 1.0), x, device=device)\n",
            "    # Simulate fused (torch.compile would actually fuse this)\n",
            "    results['unfused'].append(t_unfused)\n",
            "\n",
            "print(f'{\"Size\":>8} {\"Unfused (µs)\":>15}')\n",
            "for n, t in zip(sizes, results['unfused']):\n",
            "    print(f'{n:>8} {t:>15.1f}')\n",
            "print(f'\\nKernel launch overhead is ~constant for small tensors')\n"
        ], "c05"),
        cell_md([
            "## 2. Kernel Fusion via torch.compile\n",
            "\n",
            "PyTorch's `torch.compile` uses TorchInductor to fuse pointwise operations into a single kernel. ",
            "This eliminates intermediate memory writes and reduces kernel launch count.\n"
        ], "c06"),
        cell_code([
            "def gelu_scale_bias(x, scale, bias):\n",
            "    \"\"\"Three elementwise ops: scale, add bias, gelu.\"\"\"\n",
            "    return F.gelu(x * scale + bias)\n",
            "\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "x = torch.randn(1024 * 1024, device=device, dtype=torch.float16)\n",
            "scale = torch.tensor(2.0, device=device, dtype=torch.float16)\n",
            "bias = torch.tensor(0.5, device=device, dtype=torch.float16)\n",
            "\n",
            "# Eager (unfused)\n",
            "t_eager = benchmark_op(gelu_scale_bias, x, scale, bias, device=device)\n",
            "print(f'Eager (unfused): {t_eager:.1f} µs')\n",
            "\n",
            "if device == 'cuda':\n",
            "    try:\n",
            "        compiled = torch.compile(gelu_scale_bias, mode='reduce-overhead')\n",
            "        # Warmup compile\n",
            "        for _ in range(5):\n",
            "            compiled(x, scale, bias)\n",
            "        torch.cuda.synchronize()\n",
            "        t_compiled = benchmark_op(compiled, x, scale, bias, device=device)\n",
            "        print(f'Compiled (fused): {t_compiled:.1f} µs')\n",
            "        print(f'Speedup from fusion: {t_eager/t_compiled:.2f}x')\n",
            "    except Exception as e:\n",
            "        print(f'torch.compile: {e}')\n",
            "        print('Fusion benefit: typically 1.5-3x for pointwise ops')\n",
            "else:\n",
            "    print('Run on GPU to see fusion benefits')\n"
        ], "c07"),
        cell_md([
            "## 3. SwiGLU Fusion — A Real LLM Example\n",
            "\n",
            "SwiGLU is the FFN activation used in Llama, Mistral, and most modern LLMs. ",
            "The unfused version requires 3 kernel launches; fused version is 1.\n",
            "\n",
            "$$\\text{SwiGLU}(x) = (xW_1) \\cdot \\sigma(xW_3) \\cdot W_2$$\n"
        ], "c08"),
        cell_code([
            "class SwiGLU_Unfused(torch.nn.Module):\n",
            "    def __init__(self, d_model, d_ff):\n",
            "        super().__init__()\n",
            "        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)\n",
            "        self.w3 = torch.nn.Linear(d_model, d_ff, bias=False)\n",
            "        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)\n",
            "\n",
            "    def forward(self, x):\n",
            "        gate = F.silu(self.w1(x))   # kernel 1: silu\n",
            "        up   = self.w3(x)            # kernel 2: matmul\n",
            "        return self.w2(gate * up)    # kernel 3: elementwise mul + matmul\n",
            "\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "model = SwiGLU_Unfused(d_model=4096, d_ff=14336).to(device).half()\n",
            "x = torch.randn(1, 32, 4096, device=device, dtype=torch.float16)  # batch=1, seq=32\n",
            "\n",
            "t_unfused = benchmark_op(model, x, device=device)\n",
            "print(f'SwiGLU unfused: {t_unfused:.1f} µs (batch=1, seq=32)')\n",
            "\n",
            "if device == 'cuda':\n",
            "    try:\n",
            "        compiled_model = torch.compile(model, mode='reduce-overhead')\n",
            "        for _ in range(3): compiled_model(x)\n",
            "        torch.cuda.synchronize()\n",
            "        t_fused = benchmark_op(compiled_model, x, device=device)\n",
            "        print(f'SwiGLU compiled: {t_fused:.1f} µs')\n",
            "        print(f'Speedup: {t_unfused/t_fused:.2f}x')\n",
            "    except Exception as e:\n",
            "        print(f'Compile note: {e}')\n",
            "\n",
            "# Count theoretical kernel launches\n",
            "print('\\nKernel launch analysis:')\n",
            "print('  Unfused SwiGLU: ~5 kernel launches (2x matmul, silu, mul, matmul)')\n",
            "print('  Fused SwiGLU:   ~3 kernel launches (2x matmul, fused silu+mul)')\n",
            "print('  Benefit: 2 fewer global memory round-trips')\n"
        ], "c09"),
        cell_md([
            "## 4. FlashAttention as Kernel Fusion\n",
            "\n",
            "Standard attention requires 4 HBM round-trips (QK^T, softmax, AV, output). ",
            "FlashAttention fuses all into 1 pass using on-chip SRAM tiling — the key insight for memory-bound inference.\n"
        ], "c10"),
        cell_code([
            "# Compare memory access patterns: standard vs flash attention\n",
            "def attention_memory_accesses(seq_len, d_head, batch, num_heads, dtype_bytes=2):\n",
            "    N, d = seq_len, d_head\n",
            "    H, B = num_heads, batch\n",
            "\n",
            "    # Standard attention HBM accesses\n",
            "    qkv_read = 3 * B * H * N * d * dtype_bytes\n",
            "    attn_matrix_write = B * H * N * N * dtype_bytes   # write QK^T\n",
            "    attn_matrix_read  = B * H * N * N * dtype_bytes   # read for AV\n",
            "    output_write = B * H * N * d * dtype_bytes\n",
            "    standard_total = qkv_read + attn_matrix_write + attn_matrix_read + output_write\n",
            "\n",
            "    # FlashAttention HBM accesses (tiled, no full attn matrix)\n",
            "    flash_total = qkv_read + output_write  # only read QKV, write output\n",
            "\n",
            "    return standard_total / 1e9, flash_total / 1e9  # GB\n",
            "\n",
            "print(f'{\"Seq Len\":>8} {\"Standard (GB)\":>15} {\"Flash (GB)\":>12} {\"Reduction\":>12}')\n",
            "print('-' * 50)\n",
            "for seq in [512, 1024, 2048, 4096, 8192, 16384]:\n",
            "    std, flash = attention_memory_accesses(seq, d_head=128, batch=1, num_heads=32)\n",
            "    print(f'{seq:>8} {std:>15.3f} {flash:>12.3f} {std/flash:>11.1f}x')\n"
        ], "c11"),
        cell_md([
            "## Experiments: Try These\n",
            "\n",
            "1. **Profile with torch.profiler**: Wrap a forward pass in `torch.profiler.profile` and count kernel launches before vs after `torch.compile`.\n",
            "2. **Custom Triton kernel**: Write a fused `x * sigmoid(x)` (SiLU) Triton kernel and compare bandwidth to PyTorch eager.\n",
            "3. **Fusion sensitivity**: Measure kernel fusion benefit as a function of tensor size. At what size does the fusion benefit disappear (when memory bandwidth saturates)?\n"
        ], "c12"),
        cell_md([
            "## Key Takeaways\n",
            "\n",
            "- CUDA kernels are GPU functions; each PyTorch op dispatches at least one, incurring launch overhead and HBM round-trips.\n",
            "- Kernel fusion combines multiple pointwise ops into a single kernel, reducing global memory traffic — critical for memory-bound operations.\n",
            "- `torch.compile` with TorchInductor automatically fuses pointwise operations; the speedup is proportional to memory bandwidth saved.\n",
            "- FlashAttention is the most impactful kernel fusion in LLM inference: it eliminates the $O(N^2)$ attention matrix from HBM, enabling long-context inference.\n",
            "\n",
            "## References\n",
            "- *Inference Engineering* Ch 4.1 — Philip Kiely, Baseten Books 2026\n",
            "- Dao et al. (2022), \"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\"\n",
            "- PyTorch TorchInductor documentation\n"
        ], "c13"),
    ]
    p = write_nb(5, "cuda-kernels-fusion", cells)
    write_readme(5, "CUDA Kernels, Kernel Selection & Kernel Fusion", "Runtime",
        "Explored CUDA kernel dispatch, launch overhead, and the mechanics of kernel fusion. Benchmarked unfused vs compiled SwiGLU, quantified memory access reduction from FlashAttention-style fusion, and analyzed why fusion is transformative for memory-bound operations.",
        "Kernel fusion eliminates intermediate HBM writes. FlashAttention reduces attention memory accesses by 10-100x for long sequences by fusing QK^T, softmax, and AV into a single tiled kernel pass.",
        "cuda-kernels-fusion",
        "Memory access comparison: standard attention vs FlashAttention across sequence lengths",
        "- *Inference Engineering* Ch 4.1 (Philip Kiely, Baseten Books 2026)\n- Dao et al. (2022), \"FlashAttention\"")
    return validate(p)

print("Day 3-5 functions defined")
