# Kimi-K2 PPLX EP Correctness

> **TL;DR:** TP8/EP8 PPLX decode is now token-trace exact against the TP8/EP8
> NCCL path for the baseline probe. Clean 64-token validation on `h20-100`
> produced the same hash for both paths: `4920f088c2338236`.
>
> **Status:** Baseline fixed and committed as a correctness ground truth before
> performance work. TP1/DP8 PPLX remains a separate follow-up.

## Scope

Target comparison:

- Model: `/data/models/Kimi-K2.5`
- Machine: `h20-100`, 8x H20
- Reference path: `PEGAINFER_KIMI_PARALLEL=tp8dp1`, feature `kimi-k2`
- PPLX path: `PEGAINFER_KIMI_PARALLEL=tp8dp1`, feature `kimi-k2-pplx-ep`
- Probe: `bench_serving request --output-len {32,64} --warmup 0 --iters 1 --cuda-graph false`

The TP1/DP8 path was intentionally left out of this baseline. During debug it
matched the 12-token probe but diverged at 32 tokens, so it must not be treated
as a correctness reference yet.

## Validation Ledger

| Path | Output len | Report | Trace hash | Result |
| --- | ---: | --- | --- | --- |
| TP8 NCCL | 32 | `/tmp/kimi_nccl_tp8_clean32.json` | `6266bc659f34d5ca` | Reference |
| TP8 PPLX before final fixes | 32 | `/tmp/kimi_pplx_tp8_capacity32.json` | `6cf696c07640ef9f` | Diverged at generated token 4 |
| TP8 PPLX after routed-row weight | 32 | `/tmp/kimi_pplx_tp8_weight32.json` | `feba4dadf1fc6c22` | First boundary fixed; diverged at token 5 |
| TP8 PPLX final | 32 | `/tmp/kimi_pplx_tp8_final32.json` | `6266bc659f34d5ca` | Matches NCCL |
| TP8 NCCL clean | 64 | `/tmp/kimi_nccl_tp8_clean64.json` | `4920f088c2338236` | Reference |
| TP8 PPLX clean | 64 | `/tmp/kimi_pplx_tp8_clean64.json` | `4920f088c2338236` | Matches NCCL |

Shared 64-token prefix:

```text
[1215, 261, 5981, 14677, 1364, 91378, 2187, 924,
 276, 3628, 308, 3862, 276, 7867, 11, 996]
```

## Repro Commands

NCCL reference:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2 --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_nccl_tp8_clean64.json \
  request --output-len 64 --warmup 0 --iters 1
```

PPLX path:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_clean64.json \
  request --output-len 64 --warmup 0 --iters 1
```

Hash comparison:

```bash
uv run --no-project python - <<'PY'
import json
from pathlib import Path

paths = {
    "nccl64": Path("/tmp/kimi_nccl_tp8_clean64.json"),
    "pplx64": Path("/tmp/kimi_pplx_tp8_clean64.json"),
}
traces = {}
for name, path in paths.items():
    data = json.loads(path.read_text())
    trace = data["metrics"]["generated_token_traces"][0]
    traces[name] = trace
    print(name, trace["hash"], trace["len"], trace["prefix"])
print("match", traces["nccl64"] == traces["pplx64"])
PY
```

## Fixed Invariants

### Receive capacity

The old PPLX receive scratch used an average per-expert estimate. That is wrong
for EP all-to-all: any rank can receive skewed routes from all ranks. The bound
is now:

```text
max_routes = max_total_tokens * topk
active_experts = min(max_routes, local_experts)
capacity = max_routes + active_experts * (expert_padding - 1)
```

For decode, `max_total_tokens = local_batch * ep_world`; for prefill, it is the
synchronized prompt-token upper bound.

### Routed-row weights

NCCL applies the router top-k weight inside Marlin W2 before the BF16 expert row
is stored. PPLX must preserve the same rounding boundary:

```text
NCCL: router weight -> W2 mul_topk_weights=true -> BF16 row -> F32 sum
PPLX: router weight -> dispatch row metadata -> W2 mul_topk_weights=true -> F32 combine
```

The PPLX dispatch kernel stores `route.weight` in the 16-byte token tail after
the optional scale payload, at `token_dim + token_scale_dim`. The receive kernel
copies that value into `pplx_recv_topk_weight` using the same expert-major
padded row order as `pplx_recv_hidden`. Kimi's PPLX W2 then passes that weight
buffer with `mul_topk_weights=true`.

When `hidden_dim_scale > 0`, the scale payload remains owned by the normal
scale path; route-weight export through `out_x_scale_ptr` is only used for the
BF16 Kimi path where `hidden_dim_scale == 0`.

### F32 combine output

The final routed expert combine must not round through BF16 before the Kimi
router scale and residual/shared-expert merge. Kimi now bootstraps PPLX with
`out_dtype = F32`; `combine_recv` writes directly to `pplx_routed_f32`, uses an
element stride of `KIMI_K2_HIDDEN`, and passes dummy all-ones weights because W2
has already applied the router weight.

### No silent fallback

The `kimi-k2-pplx-ep` feature must fail startup if PPLX bootstrap fails. A
silent fallback to NCCL would make correctness probes pass for the wrong path.
The baseline run also checks for the runtime log:

```text
kimi-k2: pplx EP backends installed on all 8 ranks
```

## Rejected Attempts

| Attempt | Result | Reason |
| --- | --- | --- |
| Capacity-only fix | 32-token PPLX hash `6cf696c07640ef9f` | Necessary for skew safety, but not enough for TP8 parity. |
| Combine-side weighted rounding | First token changed to `841` in the earlier bs1 probe | Weighting after BF16 W2 output is too late to match NCCL. |
| F32 combine without routed-row W2 weights | Earlier divergence moved but did not disappear | It fixed one rounding boundary while leaving W2 weighting in the wrong place. |

## Dump Notes

A wide dump was used during the repair under `/tmp/kimi-pplx-dump/` to compare
NCCL and PPLX layer states. The dump instrumentation was removed before this
baseline: production code must not contain `dump_point`, `worker::dump`, or
`pplx_routed_out` leftovers.
