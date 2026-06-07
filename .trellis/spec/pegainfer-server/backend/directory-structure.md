# Directory Structure

> Pegainfer uses a flat Rust workspace with shared runtime/kernel crates and per-model engines; new code should land at the owner boundary, not in the server by default.

---

## Workspace Layout Contract

```text
pegainfer-server/          # binary entrypoint, CLI parsing, model selection, compatibility re-exports
pegainfer-vllm-frontend/   # Axum/vLLM protocol bridge and OpenAI-compatible routes
pegainfer-core/            # shared engine contract, runtime primitives, logging, CPU/GPU helpers
pegainfer-kernels/         # Rust kernel wrappers, CUDA FFI, build.rs CUDA/Triton build owner
pegainfer-<model>/         # per-model config, weights, executor, scheduler, tests, benches
pegainfer-comm/            # EP communication crates
kvbm/                      # Dynamo-derived KV cache crates
```

Examples:
- Workspace members are declared centrally in `Cargo.toml:1` and include `pegainfer-engine`, `pegainfer-vllm-frontend`, `pegainfer-server`, `pegainfer-core`, `pegainfer-kernels`, model crates, `pegainfer-comm`, and `kvbm/*`.
- `pegainfer-server/src/main.rs:21` owns CLI args and `pegainfer-server/src/main.rs:130` selects the model-specific engine.
- `pegainfer-vllm-frontend/src/lib.rs:108` defines LoRA routes and `pegainfer-vllm-frontend/src/lib.rs:256` bridges local vLLM engine messages into `EngineHandle`.
- `pegainfer-qwen3-4b/src/scheduler.rs:75` starts Qwen3 scheduling and `pegainfer-qwen3-4b/src/scheduler/plan.rs:55` builds the next execution plan.
- `pegainfer-kernels/src/lib.rs:4` exports first-party kernel modules; `pegainfer-kernels/build.rs:217` owns CUDA SM detection.

---

## Module Organization Rules

### Server crate

`pegainfer-server` should stay thin:
- Add CLI flags to `pegainfer-server/src/main.rs` only when they affect process startup, model loading, or served API configuration.
- Add protocol/request handling in `pegainfer-vllm-frontend`, not in `pegainfer-server`, unless it is only a compatibility re-export.
- Do not put model-specific scheduling, KV, or kernel code in `pegainfer-server`.

Current examples:
- `pegainfer-server/src/vllm_frontend.rs:1` re-exports `pegainfer_vllm_frontend::*` instead of duplicating frontend code.
- `pegainfer-server/src/lib.rs:1` re-exports `pegainfer_engine::engine::*` for compatibility.
- `pegainfer-server/src/main.rs:234` chooses between normal serving and LoRA-route serving after the engine is loaded.

### Core crate

`pegainfer-core` owns shared contracts and reusable runtime primitives:
- Engine request/event/control types live behind `pegainfer_core::engine`.
- Shared GPU/runtime helpers and ops wrappers live under flat `src/*.rs` files plus `src/<module>/` children.
- Put cross-model primitives here only when at least two model crates can use the same contract without importing a model crate.

### Model crates

Per-model crates own model-specific logic:
- `config.rs`, `weights.rs`, `executor.rs`, `scheduler.rs`, `prefill.rs`, `batch_decode.rs`, `unified_forward.rs` are model-owned.
- Scheduler submodules use the flat layout (`scheduler.rs` plus `scheduler/plan.rs`, `scheduler/effects.rs`, `scheduler/resolve.rs`), as in `pegainfer-qwen3-4b/src/scheduler.rs:8`.
- Keep request state and KV lifecycle inside the model scheduler/executor boundary.

### Kernel crate

`pegainfer-kernels` owns first-party kernel integration:
- Rust wrappers live under `pegainfer-kernels/src/ops/*.rs` and FFI declarations under `pegainfer-kernels/src/ffi/*.rs`.
- CUDA sources live under `pegainfer-kernels/csrc/shared/` for common kernels and `pegainfer-kernels/csrc/<model>/` for feature-gated model kernels.
- Build-time kernel generation and SM selection stay in `pegainfer-kernels/build.rs`; do not add a root workspace build script.

---

## Naming Conventions

- Crates use `pegainfer-<area>` or `pegainfer-<model>` names.
- Rust module files use the flat layout (`src/ops.rs` plus `src/ops/attention.rs`), never `mod.rs` for new modules.
- Model names in code should match crate identity (`Qwen3`, `Qwen35`, `KimiK2`, `DeepSeekV4`) and only the server selection layer should branch across all models.
- CUDA files are grouped by owner: `shared/*.cu` for reusable kernels, `<model>/*.cu` for model-specific kernels.

---

## Wrong vs Correct

### Wrong

```rust
// pegainfer-server/src/main.rs
// Add a Qwen-specific scheduler branch or KV admission helper here.
fn qwen3_admit_request(...) { ... }
```

### Correct

```rust
// pegainfer-qwen3-4b/src/scheduler.rs
// Keep Qwen3 admission/scheduler state in the model crate and expose only EngineHandle.
```

### Wrong

```text
pegainfer-core/src/ops/mod.rs
```

### Correct

```text
pegainfer-core/src/ops.rs
pegainfer-core/src/ops/attention.rs
```

---

## Common Mistakes

- Treating `pegainfer-server` as the owner of inference logic. It is the binary/configuration shell; model crates own execution.
- Adding protocol-specific code to a model crate. HTTP/vLLM compatibility belongs in `pegainfer-vllm-frontend`; models receive normalized engine requests.
- Moving common kernels into a model crate because only one model uses them today. Put reusable kernel wrappers in `pegainfer-kernels` and gate model-specific sources by feature when needed.
