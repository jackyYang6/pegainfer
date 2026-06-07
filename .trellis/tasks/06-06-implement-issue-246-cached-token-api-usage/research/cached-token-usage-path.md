# Research: Qwen3 cached_tokens OpenAI usage path

- **Query**: Issue #246: surface Qwen3 prefix-cache `cached_tokens` as OpenAI `/v1/completions` `usage.prompt_tokens_details.cached_tokens`, frontend pass-through only.
- **Scope**: mixed (local repo + vendored/upstream vLLM cache)
- **Date**: 2026-06-07

## Findings

### Files Found

| File Path | Description |
|---|---|
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-engine/src/engine.rs` | Shared `TokenEvent::Scheduled { prompt_tokens, cached_tokens }` contract. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs` | `PrefillRequestResult.cached_tokens` documents prefix-cache KV reuse count. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/resolve.rs` | Copies executor `result.cached_tokens` into `ScheduledEffect`. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/effects.rs` | Emits `TokenEvent::Scheduled { cached_tokens: scheduled.cached_tokens }`. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs` | Local Rust vLLM bridge converts `TokenEvent::Scheduled` into `PrefillStats` and currently also patches final OpenAI usage JSON. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-sim/src/lib.rs` | Simulated engine can emit configured `scheduled_cached_tokens` for HTTP e2e coverage. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-sim/tests/frontend_e2e.rs` | Current e2e tests assert cached tokens in non-streaming and streaming usage chunks. |
| `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/rust/src/server/src/routes/openai/completions.rs` | Vendored/upstream Rust OpenAI completions constructs usage counts with `Usage::from_counts`; no cached-token bridge. |
| `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/rust/src/server/src/routes/openai/utils/types.rs` | Rust `Usage` type has optional `prompt_tokens_details`, but `from_counts` leaves it `None`. |
| `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/entrypoints/openai/completion/serving.py` | Python vLLM intended behavior: copy `RequestOutput.num_cached_tokens` into `PromptTokenUsageInfo.cached_tokens` when prompt-token-details are enabled and value is non-zero. |
| `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/v1/engine/output_processor.py` | Python vLLM bridge from `EngineCoreOutput.prefill_stats.num_cached_tokens` to `RequestOutput.num_cached_tokens`. |
| `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/v1/core/sched/scheduler.py` | Python scheduler fills `request.prefill_stats` from local/external cached token counts. |

### Code Patterns

Local engine/scheduler path already carries a scheduler-computed value, not a frontend guess:

- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-engine/src/engine.rs:132-138` defines `TokenEvent::Scheduled { queued_at_unix_s, scheduled_at_unix_s, prompt_tokens, cached_tokens }`.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs:367-375` defines `PrefillRequestResult.cached_tokens` as “Prompt tokens served from the prefix cache (KV reused, not recomputed).”
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/resolve.rs:39-44` sets `ScheduledEffect.cached_tokens = result.cached_tokens`.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/effects.rs:84-94` sends that value in `TokenEvent::Scheduled`.

Local frontend bridge path to `EngineCoreOutput.prefill_stats` exists:

- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:935-983` handles `TokenEvent::Scheduled`, creates `PrefillStats { num_prompt_tokens, num_computed_tokens, num_cached_tokens, num_local_cached_tokens, num_external_cached_tokens }`, and stores the same cached count in an external-request-id map.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:995-1002` attaches `first_token_prefill_stats.take()` to the first token output.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:1057-1077` passes `prefill_stats` into `engine_output`.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:1147-1168` sets `EngineCoreOutput.prefill_stats`.

Local frontend currently has a JSON patcher around vLLM Rust server output:

- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:54-55` declares global `CACHED_TOKENS_BY_REQUEST_ID`.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:575-584` wraps `/v1/completions` with `cached_token_usage_routes`.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:657-684` forwards the request to the vLLM router and dispatches to non-streaming or streaming patching.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:687-692` detects streaming by checking the URL query for exactly `stream=true`; OpenAI completions usually carry `stream` in the JSON body, so this detector is likely not aligned with normal request shape.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:694-728` non-stream patch reads the JSON response `id`, removes cached count from the map, and calls `patch_usage_value`.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:731-799` streaming patch rewrites SSE chunks that have non-null `usage`.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:802-819` creates/overwrites `usage.prompt_tokens_details.cached_tokens`.

Vendored/upstream Rust vLLM usage construction does not consume `prefill_stats`:

- Revision: `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c` at commit `4aaed4ca225a3745aa1e18864dad0599d3ac7626` (`git describe`: `4aaed4ca2`).
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/rust/src/server/src/routes/openai/utils/types.rs:312-330` defines `Usage { prompt_tokens, total_tokens, completion_tokens, prompt_tokens_details }` and `Usage::from_counts`, which sets `prompt_tokens_details: None`.
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/rust/src/server/src/routes/openai/completions.rs:179-197` non-streaming response uses `Usage::from_counts(collected.prompt_token_ids.len(), collected.token_ids.len())`; `prompt_tokens` is prompt token id count, `completion_tokens` is generated token id count, `total_tokens` is their sum.
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/rust/src/server/src/routes/openai/completions.rs:297-306` streaming `include_usage` final chunk uses `Usage::from_counts(finished.prompt_token_count, finished.output_token_count)`.
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/rust/src/server/src/routes/openai/completions/convert.rs:58-60` derives `include_usage` from `request.stream_options.include_usage`, not query params.
- `rg` over vendored Rust server found `prefill_stats` only in protocol/tests, not in OpenAI completions usage construction. The Rust server has the field type but no bridge analogous to Python.

Upstream Python vLLM intended source and attachment:

- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/v1/core/sched/scheduler.py:654-661` scheduler records prefill stats on first scheduled prefill: `request.prefill_stats.set(num_prompt_tokens=..., num_local_cached_tokens=..., num_external_cached_tokens=...)`.
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/v1/engine/output_processor.py:628-633` on first prefill output copies `engine_core_output.prefill_stats.num_cached_tokens` into `req_state.num_cached_tokens`.
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/v1/engine/output_processor.py:363-373` attaches `num_cached_tokens=req_state.num_cached_tokens` to `RequestOutput`.
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/entrypoints/openai/completion/serving.py:309-311` streaming completions read `res.num_cached_tokens` on first iteration.
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/entrypoints/openai/completion/serving.py:440-449` streaming final usage chunk sets `PromptTokenUsageInfo(cached_tokens=num_cached_tokens)` only when `enable_prompt_tokens_details` is true and `num_cached_tokens` is truthy/non-zero.
- `/Users/jacky/.cargo/git/checkouts/vllm-374ebce006d85a6a/4aaed4c/vllm/entrypoints/openai/completion/serving.py:577-590` non-streaming usage does the same from `last_final_res.num_cached_tokens`.
- Therefore intended source is request output metadata that was populated from `EngineCoreOutput.prefill_stats`, which itself originates in scheduler prefill stats. It is not recomputed at the HTTP layer.

### Related Specs

No specific `.trellis/spec/**/*.md` contract for cached-token usage was found in the searched output beyond package structure. Relevant project package areas are `pegainfer-server`/backend and frontend bridge code.

## Conclusions / Recommended Next Implementation Step

1. `EngineCoreOutput.prefill_stats.num_cached_tokens` already has a local path up to the first token output in `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs`, and Qwen3 emits scheduler-derived cached counts through `TokenEvent::Scheduled`.
2. The missing bridge is in vendored/upstream Rust vLLM OpenAI completions: `Usage::from_counts` and callers never read `prefill_stats`/request-output cached metadata, unlike Python vLLM. Local code currently compensates with a frontend response/SSE JSON patcher.
3. Smallest pass-through-only local change depends on whether editing vendored dependency is feasible. Within current repo only, the minimal viable bridge is the existing `/v1/completions` wrapper/map patch path, but it should key off the JSON request body for `stream`/`stream_options.include_usage` if continuing to patch streaming output; do not recompute from prompt length. A cleaner dependency-level change would be adding cached-token metadata to the Rust server text output/finished metadata and setting `Usage.prompt_tokens_details`, matching Python, but that is not “frontend pass-through only.”
4. To avoid changing unrelated issue #78 streaming accounting, only attach `prompt_tokens_details.cached_tokens` to the already-existing final `usage` object/chunk; do not alter `prompt_tokens`, `completion_tokens`, `total_tokens`, continuous usage, or whether a usage chunk is emitted.

## Test Scope Guidance

- Keep/add frontend unit coverage in `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs` for `TokenEvent::Scheduled.cached_tokens -> EngineCoreOutput.prefill_stats.num_cached_tokens`; this is valid pass-through coverage.
- Keep/add sim HTTP e2e coverage in `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-sim/tests/frontend_e2e.rs` for non-streaming final JSON and streaming final `include_usage` chunk; this validates the actual OpenAI surface without GPU.
- Qwen3 scheduler unit coverage is valid if it asserts the scheduler/executor result’s `cached_tokens` is forwarded in `TokenEvent::Scheduled`; it should not assert OpenAI JSON because that is frontend/server responsibility.
- GPU Qwen3 manual/e2e is useful as optional verification that real prefix-cache hits produce non-zero cached counts, but overreach for a small frontend pass-through change if the scheduler unit and sim HTTP tests cover the data path. Do not require GPU tests for unrelated streaming usage accounting.

## Caveats / Not Found

- The local working tree already contains modifications related to this issue; observations reflect current files, not necessarily `HEAD`.
- The vendored Rust vLLM server at commit `4aaed4ca225a3745aa1e18864dad0599d3ac7626` has a `PromptTokenUsageInfo` type but no found code path that fills it for completions.
- The local wrapper’s `request_is_streaming_completion` currently checks URL query `stream=true`; the sim tests send `stream` in the JSON body. If this remains unchanged, normal streaming requests may be routed through the non-stream body patch path rather than the SSE patch path.
