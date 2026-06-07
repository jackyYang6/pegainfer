# Pegainfer Backend Code-Spec Index

> Executable conventions for pegainfer's Rust/CUDA inference engine, server entrypoint, vLLM frontend bridge, per-model schedulers, and kernel crate.

---

## Scope

This spec package is named `pegainfer-server` because it is the default Trellis package, but agents should treat it as the pegainfer workspace code-spec for first-party code: `pegainfer-server`, `pegainfer-vllm-frontend`, `pegainfer-core`, `pegainfer-kernels`, and per-model crates such as `pegainfer-qwen3-4b` and `pegainfer-qwen35-4b`.

Do not apply these rules to vendored third-party trees unless the task explicitly touches first-party integration code around them.

---

## Guidelines Index

| Guide | What it controls | Status |
|-------|------------------|--------|
| [Directory Structure](./directory-structure.md) | Workspace/crate/module layout and where new code belongs | Filled |
| [Database Guidelines](./database-guidelines.md) | No database/ORM layer; state ownership and persistence boundaries | Filled |
| [Error Handling](./error-handling.md) | `anyhow` propagation, CLI parse errors, HTTP error bodies, scheduler request failures | Filled |
| [Quality Guidelines](./quality-guidelines.md) | Release builds, test gates, CUDA/accuracy expectations, forbidden shortcuts | Filled |
| [Logging Guidelines](./logging-guidelines.md) | `log`/`logforth` initialization, levels, noisy module filters, sensitive payload boundaries | Filled |

---

## High-Level Contract

- The HTTP/vLLM-facing layer submits `GenerateRequest` values through `EngineHandle`; model crates own scheduling, execution, KV state, and tests.
- `pegainfer-kernels/build.rs` is the build owner for CUDA, Triton AOT, and feature-gated model kernels; the workspace root has no package build script.
- Model crates should hide architecture-specific execution details behind their `start_engine`/scheduler/executor boundary instead of leaking model state into `pegainfer-server`.
- Use `--release` for build/test/run commands that execute CUDA or model paths.

---

**Language**: All code-spec documentation is written in English.
