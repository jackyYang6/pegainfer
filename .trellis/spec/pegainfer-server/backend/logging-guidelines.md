# Logging Guidelines

> Pegainfer initializes unified `log`/`logforth` logging once, uses concise lifecycle/performance logs, warns on recoverable request/bridge failures, and avoids logging raw prompts or secrets.

---

## Scope / Trigger

Use this spec when touching:
- process startup or logging initialization
- scheduler lifecycle/failure handling
- model loading and build/runtime diagnostics
- frontend bridge request handling
- benchmark/reporting code

---

## Logging Setup Contract

- Initialize logging once at process startup through `pegainfer::logging::init_default()`.
- Use the `log` facade (`info!`, `warn!`, `debug!`, `error!`) in first-party code unless a crate has a specific tracing integration.
- Respect `RUST_LOG`; default noisy HTTP stack modules are clamped to `warn` when global logging is verbose.

Examples:
- `pegainfer-server/src/main.rs:88` calls `logging::init_default()` before startup logs.
- `pegainfer-core/src/logging.rs:11` uses `Once` to guard initialization.
- `pegainfer-core/src/logging.rs:28` defines noisy module defaults for `h2`, `hyper`, `hyper_util`, `axum`, and `tower`.
- `pegainfer-core/src/logging.rs:77` resolves `RUST_LOG` overrides.
- `pegainfer-core/src/logging.rs:119` tests filter behavior.

---

## Log Levels

| Level | Use for | Examples |
|-------|---------|----------|
| `info` | lifecycle milestones, selected config, engine loaded, scheduler ready, adapter control operations, benchmark/report output | `pegainfer-server/src/main.rs:115`, `pegainfer-qwen3-4b/src/scheduler.rs:143`, `pegainfer-qwen3-4b/src/scheduler.rs:368` |
| `warn` | recoverable failures where the process/request loop continues | `pegainfer-qwen3-4b/src/scheduler.rs:201`, `pegainfer-vllm-frontend/src/lib.rs:328` |
| `debug` | detailed GPU/model loading/capture diagnostics that are useful under `RUST_LOG=debug` but too noisy by default | `pegainfer-core/src/cuda_graph.rs:62`, `pegainfer-qwen3-4b/src/weights.rs:102` |
| `error` | background/reporting failures that cannot be returned to a caller | `pegainfer-server/src/trace_reporter.rs:108` |

---

## What to Log

- Startup model type and runtime options that affect execution behavior. Existing example: `pegainfer-server/src/main.rs:118` logs model path, requested/effective CUDA graph, LoRA, device, TP/DP, and EP backend.
- Engine load completion with elapsed time. Examples: `pegainfer-server/src/main.rs:146`, `pegainfer-server/src/main.rs:210`, `pegainfer-server/src/main.rs:228`.
- Scheduler readiness and shutdown because these identify GPU worker lifecycle. Examples: `pegainfer-qwen3-4b/src/scheduler.rs:143` and `pegainfer-qwen3-4b/src/scheduler.rs:164`.
- Recoverable execution failures with enough context to diagnose the class of failure.
- Build-time warnings from `build.rs` using Cargo warning output, not runtime logging. Examples: invalid SM token warnings in `pegainfer-kernels/build.rs:224` and GPU detection fallback in `pegainfer-kernels/build.rs:243`.

---

## What NOT to Log

- Do not log raw prompts, generated text, token arrays, API bodies, model secrets, or full filesystem contents.
- Do not log LoRA adapter weights or file contents. Adapter names and paths may be logged for control-plane diagnostics, as in `pegainfer-qwen3-4b/src/scheduler.rs:368`, but avoid expanding or reading sensitive files.
- Do not emit per-token `info` logs in hot decode loops. Use debug-only traces, structured benchmark output, or dedicated tracing/report files.
- Do not turn noisy HTTP internals to debug by default; `RUST_LOG` can opt in and the default filter suppresses common noisy modules.

---

## Structured / Diagnostic Logging

- Runtime logs are text-oriented through `logforth::layout::TextLayout`; do not introduce a second logger in a crate.
- For performance/trace artifacts, prefer explicit report files or trace reporter output instead of stuffing large data into logs.
- If a background task cannot return an error to a caller, log with `error!` and include the target path or operation, as in `pegainfer-server/src/trace_reporter.rs:108`.

---

## Good / Base / Bad Cases

- Good: `info!` once per major lifecycle event and `warn!` for recoverable request failures.
- Base: `debug!` for optional CUDA Graph capture details.
- Bad: `info!` for every generated token, raw request body, or full prompt token list.

---

## Tests Required

- Logging filter changes require unit tests like `pegainfer-core/src/logging.rs:123`.
- Startup option logs should be manually checked during a release smoke run when CLI behavior changes.
- Changes that create report/trace files should test or manually verify IO failure handling.

---

## Wrong vs Correct

### Wrong

```rust
info!("prompt={:?} tokens={:?}", prompt, prompt_tokens);
```

### Correct

```rust
info!("Runtime options: model_path={}, requested_cuda_graph={}, effective_cuda_graph={}", ...);
```

### Wrong

```rust
error!("Execution step failed: {e}");
return;
```

for a recoverable per-request scheduler failure.

### Correct

```rust
warn!("Execution step failed: {e}");
fail_touched_requests(...);
continue;
```
