# Quality Guidelines

> Pegainfer quality is guarded by release-mode Rust checks, model-specific GPU/accuracy tests where applicable, and explicit contract tests for scheduler/frontend behavior.

---

## Scope / Trigger

Use this spec for any code change under first-party pegainfer crates. Apply extra rigor when touching:
- CUDA kernels, CUDA Graph, FlashInfer/cuBLAS wrappers, or `pegainfer-kernels/build.rs`
- scheduler admission, request lifecycle, KV/prefix cache, or sampling behavior
- OpenAI/vLLM protocol compatibility
- model correctness/accuracy paths
- build feature gates or environment-variable wiring

---

## Required Patterns

### Always use release mode for meaningful validation

Debug builds are too slow for GPU/CUDA paths and can time out. Use release mode for build/run/test commands that execute the engine or kernels.

Canonical commands from project instructions:

```bash
cargo test --release --workspace --lib
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test hf_golden_gate
PEGAINFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test hf_golden_gate
PEGAINFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test e2e_scheduler
```

### Match the test to the change surface

| Change surface | Minimum validation |
|----------------|--------------------|
| Pure helper/parser logic | focused unit tests |
| CLI or env parsing | parser unit tests plus startup smoke if feasible |
| vLLM/OpenAI frontend | route/protocol tests or simulated-engine serving coverage |
| scheduler/admission/KV | scheduler unit/integration tests and relevant model e2e |
| model logits/sampling/correctness | HF/vLLM golden gate for that model |
| CUDA kernel wrapper | Rust wrapper test plus GPU kernel test/bench if available |
| build.rs / CUDA build env | hardware-independent parser tests plus release build |

Examples:
- LoRA CLI parser tests live next to the parser in `pegainfer-server/src/main.rs:337`.
- Qwen3 scheduler planning tests live in `pegainfer-qwen3-4b/src/scheduler/plan.rs:244`.
- Qwen3 HF-golden methodology is documented in `pegainfer-qwen3-4b/tests/hf_golden_gate.rs:1`.
- Logging filter behavior is unit-tested in `pegainfer-core/src/logging.rs:119`.

### Accuracy gates compare distributions, not brittle text snapshots

For logits/correctness work, prefer teacher-forced golden comparisons with structural checks over exact free-text snapshots.

Existing Qwen3 gate contract:
- replay fixed HF sequences through pegainfer (`pegainfer-qwen3-4b/tests/hf_golden_gate.rs:11`)
- enforce argmax regret where HF has a clear winner (`pegainfer-qwen3-4b/tests/hf_golden_gate.rs:20`)
- assert mean and p99 logprob deltas, report but do not assert absolute max (`pegainfer-qwen3-4b/tests/hf_golden_gate.rs:24`)
- cover bs=1 sequential, batched eager, and batched CUDA graph paths (`pegainfer-qwen3-4b/tests/hf_golden_gate.rs:30`)

---

## Forbidden Patterns

- Do not run or report debug-mode GPU/model tests as sufficient validation.
- Do not replace accuracy gates with exact generated text unless the task specifically concerns text-level API behavior.
- Do not skip hook/test failures with `--no-verify` or equivalent bypasses.
- Do not add feature flags or compatibility shims just to avoid updating call sites.
- Do not add speculative abstractions around a one-model implementation unless the same contract is already needed by multiple models.
- Do not silently ignore unsupported API parameters; either honor them, reject them, or document the explicit compatibility boundary.
- Do not assert hardware-unstable single worst logprob deltas when mean/p99/regret are the intended correctness contract.

---

## Validation & Error Matrix

| Risk | Required guard |
|------|----------------|
| GPU code compiles but wrong architecture target | `PEGAINFER_CUDA_SM` parsing/build coverage or real release build on target host |
| Scheduler drops requests under pressure | admission/deferred/rejection tests |
| CUDA graph path corrupts padding slots | batched graph correctness gate covering bucket straddles |
| Sampling parameter silently ignored | request-to-sampling conversion test plus API/e2e behavior test |
| Token/logprob regression hidden by free-greedy divergence | teacher-forced golden logits replay |
| Frontend overhead/perf claim based on direct bench | HTTP/simulated serving benchmark evidence |

---

## Good / Base / Bad Cases

- Good: a scheduler change includes a unit test for the planning/admission edge case and runs the relevant model e2e or golden gate.
- Base: a pure CLI parser change includes focused parser tests and `cargo test --release -p pegainfer-server`.
- Bad: a CUDA wrapper change is marked done after `cargo check` only, without release build or GPU validation on the touched path.

---

## Tests Required

Before reporting completion, state which of these were run or why they could not run:
- `cargo test --release --workspace --lib` for broad Rust library coverage.
- Package-specific release tests for changed crates.
- GPU/model-weight gates when touching execution, logits, sampling, KV, or CUDA kernels.
- Serving/HTTP smoke when touching `pegainfer-vllm-frontend` or OpenAI-compatible behavior.
- Bench/profiling only when making performance claims; do not infer serving performance from direct in-process benches alone.

---

## Wrong vs Correct

### Wrong

```bash
cargo test -p pegainfer-qwen3-4b --test hf_golden_gate
```

reported as the CUDA correctness gate.

### Correct

```bash
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test hf_golden_gate
```

### Wrong

```rust
// Assert the absolute maximum logprob delta across every sampled token.
assert!(max_delta < fixed_limit);
```

### Correct

```rust
// Assert regret + mean + p99; report absolute max for diagnostics only.
```

---

## Scenario: Engine scheduled cached-token API usage

### 1. Scope / Trigger

- Trigger: cross-layer request metadata flows from model scheduler/executor through `TokenEvent` into OpenAI/vLLM-compatible usage fields.
- Applies when changing `TokenEvent::Scheduled`, scheduler request lifecycle metadata, or frontend usage accounting.

### 2. Signatures

```rust
// pegainfer-engine/src/engine.rs
TokenEvent::Scheduled {
    queued_at_unix_s: f64,
    scheduled_at_unix_s: f64,
    prompt_tokens: usize,
    cached_tokens: usize,
}
```

```rust
// pegainfer-vllm-frontend/src/lib.rs
PrefillStats {
    num_prompt_tokens,
    num_computed_tokens,
    num_cached_tokens,
    num_local_cached_tokens,
    num_external_cached_tokens,
}
```

### 3. Contracts

- `prompt_tokens` is the original prompt token count for the request.
- `cached_tokens` is engine-reported per-request prefix-cache reuse; the frontend must not infer or recompute it.
- Non-reporting engines must send `cached_tokens: 0` so old behavior remains explicit.
- Qwen3 must preserve executor `PrefillRequestResult.cached_tokens` through scheduler resolve/effects before emitting `TokenEvent::Scheduled`.
- Qwen3 may emit `Scheduled` after executor prefix matching, but it must use the recorded prefill-selection timestamp and deliver the event before any `Token` or `Finished` event for that request.
- Frontend usage mapping is:
  - `num_prompt_tokens = prompt_tokens`
  - `num_computed_tokens = prompt_tokens.saturating_sub(cached_tokens)`
  - `num_cached_tokens = cached_tokens`
  - `num_local_cached_tokens = cached_tokens`
  - `num_external_cached_tokens = 0`

### 4. Validation & Error Matrix

| Condition | Required behavior |
|-----------|-------------------|
| Engine has no prefix-cache accounting | Send `cached_tokens: 0`; API usage remains cold/zero. |
| Qwen3 executor reports a cache hit | Preserve the count into `Scheduled.cached_tokens`; API usage reports nonzero cached tokens. |
| `cached_tokens > prompt_tokens` from an engine bug | Frontend uses saturating subtraction for computed tokens and does not panic. |
| Request finishes during prefill or max-tokens-zero path | Emit `Scheduled` before `Finished` so usage metadata is available. |
| Streaming response with usage chunk | Patch/use the same scheduled cached-token metadata as non-streaming; do not duplicate cache logic. |

### 5. Good/Base/Bad Cases

- Good: Qwen3 warm repeated prompt produces a `Scheduled` event with nonzero `cached_tokens`, followed by token/finish events, and OpenAI usage reports the same cached-token count.
- Base: DeepSeek/Kimi/simulated engines that do not compute cache hits emit `cached_tokens: 0` and preserve existing API usage.
- Bad: frontend hard-codes cached-token fields to zero or estimates cache hits from prompt length.

### 6. Tests Required

- Unit-test frontend `TokenEvent::Scheduled` to `PrefillStats` pass-through, including `num_computed_tokens` saturation.
- Simulated-engine HTTP tests should cover non-streaming and streaming usage with configured scheduled cached tokens.
- Qwen3 scheduler tests should assert executor cached-token propagation and `Scheduled` ordering before first `Token` or `Finished`.
- GPU/model-weight validation is required before making an end-to-end Qwen3 cold/warm API claim.

### 7. Wrong vs Correct

#### Wrong

```rust
PrefillStats {
    num_prompt_tokens: prompt_tokens as u32,
    num_computed_tokens: prompt_tokens as u32,
    num_cached_tokens: 0,
    num_local_cached_tokens: 0,
    num_external_cached_tokens: 0,
}
```

#### Correct

```rust
PrefillStats {
    num_prompt_tokens: prompt_tokens as u32,
    num_computed_tokens: prompt_tokens.saturating_sub(cached_tokens) as u32,
    num_cached_tokens: cached_tokens as u32,
    num_local_cached_tokens: cached_tokens as u32,
    num_external_cached_tokens: 0,
}
```

---

## Code Review Checklist

- Is the change in the owning crate/module per directory-structure spec?
- Are boundary errors mapped explicitly and tested?
- Are unsupported API parameters rejected or honored instead of ignored?
- Did the validation use release mode for GPU/model paths?
- If correctness changed, does the test compare the right invariant rather than a brittle artifact?
- If performance is claimed, is it backed by an appropriate benchmark path and documented command?
