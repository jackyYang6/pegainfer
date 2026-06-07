# Error Handling

> Pegainfer uses `anyhow` for internal fallible Rust paths, typed errors at public/control boundaries, explicit HTTP status mapping at the frontend, and per-request `TokenEvent` failures inside schedulers.

---

## Scope / Trigger

Use this spec whenever a change touches:
- CLI parsing or model startup
- OpenAI/vLLM HTTP routes
- `EngineHandle`, `GenerateRequest`, `TokenEvent`, or control-plane requests
- scheduler admission/execution failure behavior
- CUDA/build/runtime wrappers that convert low-level errors into Rust errors

---

## Signatures

### Process startup

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()>;
```

Examples:
- `pegainfer-server/src/main.rs:87` returns `anyhow::Result<()>`.
- `pegainfer-server/src/main.rs:92` wraps model detection with `.with_context(...)`.
- `pegainfer-server/src/main.rs:98` rejects invalid CLI combinations with `bail!(...)`.

### CLI value parser

```rust
fn parse_lora_modules_arg(value: &str) -> Result<LoraModule, String>;
```

`clap` value parsers return `Result<T, String>`, not `anyhow::Result<T>`, so the error text can be surfaced as argument validation. See `pegainfer-server/src/main.rs:270`.

### HTTP route errors

HTTP handlers return `axum::response::Response` and map errors at the boundary:

```rust
#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}
```

Examples:
- `pegainfer-vllm-frontend/src/lib.rs:89` defines the body.
- `pegainfer-vllm-frontend/src/lib.rs:136` validates LoRA load requests.
- `pegainfer-vllm-frontend/src/lib.rs:168` maps `EngineControlError::Unsupported` to `404`.
- `pegainfer-vllm-frontend/src/lib.rs:175` maps channel closure to `503`.
- `pegainfer-vllm-frontend/src/lib.rs:182` maps operation failure to `400`.

### Scheduler request failures

Schedulers keep the process alive on per-step failures and fail touched requests rather than crashing the scheduler thread:

```rust
match execute_plan(...) {
    Ok(v) => v,
    Err(e) => {
        warn!("Execution step failed: {e}");
        fail_touched_requests(..., &e.to_string());
        continue;
    }
}
```

Qwen3 examples: `pegainfer-qwen3-4b/src/scheduler.rs:198` and `pegainfer-qwen3-4b/src/scheduler.rs:319`.

---

## Contracts

### Internal Rust errors

- Use `anyhow::Result<T>` for internal fallible operations where callers only need context and propagation.
- Add `.context(...)` / `.with_context(...)` at subsystem boundaries: model detection, engine start, IPC connect/send/receive, file loading, and CUDA wrapper calls.
- Use `bail!` / `ensure!` for explicit invariant validation in fallible functions.

### Boundary errors

- CLI parsing functions used by `clap` return `Result<T, String>`.
- HTTP endpoints return `(StatusCode, Json(ErrorBody))` or a text success body as already established.
- Engine control uses typed `EngineControlError` at the boundary and converts to HTTP status only in `pegainfer-vllm-frontend`.
- Generation failures should flow to request streams through `TokenEvent::Error` / rejection events rather than panicking.

---

## Validation & Error Matrix

| Condition | Boundary | Response |
|-----------|----------|----------|
| `--lora-modules` without `--enable-lora` | CLI startup | `bail!("--lora-modules requires --enable-lora")` |
| `--enable-lora` with non-Qwen3 model | CLI startup | `bail!("--enable-lora is currently supported only for Qwen3")` |
| Empty LoRA name/path in CLI parser | CLI parser | `Err("--lora-modules name/path must not be empty")` |
| Empty `lora_name` or `lora_path` in HTTP request | HTTP route | `400 {"error": "... must not be empty"}` |
| Unsupported LoRA control operation | HTTP route | `404 {"error": ...}` |
| Engine control channel closed | HTTP route | `503 {"error": ...}` |
| Scheduler execution step fails | Scheduler | warn log + fail touched requests + continue loop |
| Malformed local vLLM engine frame | Local bridge | `anyhow::bail!` from bridge handler; caller logs warning |

---

## Good / Base / Bad Cases

- Good: add context when crossing a subsystem boundary, e.g. `detect_model_type(...).with_context(|| format!("failed to detect model type from {}", path.display()))` as in `pegainfer-server/src/main.rs:92`.
- Base: propagate low-level helper failures with `?` inside the same subsystem when no extra context is useful.
- Bad: panic on user input, HTTP request validation, missing model path, or scheduler per-request failures.

---

## Tests Required

- CLI/parser changes: unit-test valid forms and invalid forms next to the parser. Existing LoRA parser tests start at `pegainfer-server/src/main.rs:337`.
- HTTP/control changes: test status code and `ErrorBody.error` for each mapped failure class.
- Scheduler admission/execution changes: test rejection/error events and that unrelated active requests are not failed.
- CUDA/build wrappers: test parse/normalization helpers when hardware-independent; require release GPU tests for execution paths.

---

## Wrong vs Correct

### Wrong

```rust
if request.lora_name.is_empty() {
    panic!("missing lora_name");
}
```

### Correct

```rust
if request.lora_name.is_empty() {
    return bad_request("lora_name must not be empty");
}
```

### Wrong

```rust
let handle = start_engine(path, options)?;
```

at a boundary where the caller loses model context.

### Correct

```rust
let handle = start_engine(path, options).context("failed to start Qwen3 engine")?;
```

---

## Common Mistakes

- Returning `anyhow::Error` directly from HTTP handlers. Map it to a stable HTTP status/body at the frontend boundary.
- Using `expect`/`unwrap` on user-controlled input. Reserve `expect` for impossible process setup failures or tests.
- Failing the whole scheduler on one request's execution error. The established scheduler behavior is to warn, notify affected requests, and continue.
