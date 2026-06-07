# State and Persistence Guidelines

> Pegainfer has no application database or ORM layer; durable model artifacts live on disk, runtime state lives in schedulers/executors/KV managers, and API state is in-memory unless an integration explicitly defines otherwise.

---

## Scope / Trigger

This file replaces generic database guidance for pegainfer. Use it when a task touches:
- request state, scheduler queues, KV cache/block allocation, prefix cache, or LoRA adapter state
- model weight loading and filesystem artifacts
- benchmark/report output files
- any proposed database, migration, or persistent service integration

---

## Current Contract

- There is no SQL database, ORM, migration system, or table/column naming convention in first-party pegainfer serving code.
- Model weights, tokenizer/config files, golden fixtures, benchmark outputs, and trace/report artifacts are filesystem-backed.
- Runtime request state is owned by scheduler/executor data structures, not by global mutable storage.
- API-visible LoRA adapter lists are in-memory and mirrored around engine control operations.

Examples:
- `pegainfer-vllm-frontend/src/lib.rs:52` keeps LoRA route state as `EngineHandle` plus `Arc<RwLock<HashSet<String>>>`.
- `pegainfer-vllm-frontend/src/lib.rs:161` inserts an adapter name only after engine load succeeds.
- `pegainfer-vllm-frontend/src/lib.rs:206` removes an adapter name only after engine unload succeeds.
- `pegainfer-qwen3-4b/src/scheduler.rs:137` keeps active request state in a scheduler-owned `Vec<ActiveRequestState>`.
- `pegainfer-qwen3-4b/src/scheduler.rs:141` keeps KV-pressure-deferrable requests in `Vec<PendingRequest>`.
- `pegainfer-core/src/weight_loader.rs` loads safetensor shards from the model directory instead of a database.

---

## State Ownership Rules

### Request state

- Keep per-request generation state inside the model scheduler/executor.
- Submit normalized `GenerateRequest` values through `EngineHandle`; do not persist requests in the frontend.
- If KV pressure prevents immediate admission, defer in the scheduler and re-evaluate rather than dropping or externally storing requests.

### Control-plane state

- Control APIs should update in-memory frontend state only after the underlying engine operation succeeds.
- Keep control request ordering explicit. Qwen3 LoRA control queues control commands and applies them when idle, as in `pegainfer-qwen3-4b/src/scheduler.rs:222` and `pegainfer-qwen3-4b/src/scheduler.rs:242`.

### Filesystem artifacts

- Model artifacts are read from `--model-path` and should be validated at load time.
- Reports/traces may write files, but must treat IO failures as errors or logs, not silently succeed.
- Tests that require model weights should skip clearly when the model path is absent; see `pegainfer-qwen3-4b/tests/hf_golden_gate.rs:108`.

---

## If a Persistent Store Is Proposed

A task that adds a real database or external persistent service must first create/update a code-spec section with all seven contract sections:
1. Scope / Trigger
2. Signatures
3. Contracts
4. Validation & Error Matrix
5. Good/Base/Bad Cases
6. Tests Required
7. Wrong vs Correct

Minimum contract fields:
- connection/env keys
- schema or key shape
- ownership of writes
- failure behavior when the store is unavailable
- migration or compatibility plan
- tests that run without production credentials

---

## Wrong vs Correct

### Wrong

```rust
static mut ACTIVE_REQUESTS: Vec<RequestState> = Vec::new();
```

### Correct

```rust
fn scheduler_loop(...) {
    let mut active: Vec<ActiveRequestState> = Vec::new();
    let mut deferred: Vec<PendingRequest> = Vec::new();
}
```

### Wrong

```rust
state.adapter_names.write().await.insert(name.clone());
state.handle.load_lora_adapter(request).await?;
```

### Correct

```rust
state.handle.load_lora_adapter(request).await?;
state.adapter_names.write().await.insert(name.clone());
```

---

## Common Mistakes

- Inventing a database abstraction for request/adapter state. The current serving contract is in-memory and scheduler-owned.
- Updating frontend-visible state before engine control succeeds.
- Treating golden fixtures or model weights as mutable runtime state. They are filesystem inputs for validation/loading.
