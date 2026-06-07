# Implement issue 246 cached-token API usage

## Goal

Fix GitHub issue #246 so Qwen3 prefix-cache hits propagate from the Qwen3 executor to OpenAI/vLLM-compatible API usage fields instead of always reporting zero cached tokens.

## Context

Issue #247 has already been implemented and merged separately via PR #283. This task is only for #246.

Latest aligned #246 finding:

* `pegainfer-engine/src/engine.rs` defines `TokenEvent::Scheduled` with queue time, schedule time, and `prompt_tokens`, but no cached-token metadata.
* `pegainfer-vllm-frontend/src/lib.rs` builds `PrefillStats` from `TokenEvent::Scheduled` and currently hard-codes cached-token fields to zero.
* Qwen3 executor computes `PrefillRequestResult.cached_tokens` in `pegainfer-qwen3-4b/src/executor.rs`.
* Qwen3 scheduler does not currently emit `TokenEvent::Scheduled`, and `scheduler/resolve.rs` drops `PrefillRequestResult.cached_tokens` when converting prefill results into effects.

Execution chain:

```text
HTTP /v1/completions
  -> pegainfer-server
  -> pegainfer-vllm-frontend: text/API bridge, tokenization/detokenization glue, response usage
  -> pegainfer-engine: GenerateRequest / TokenEvent contract
  -> pegainfer-qwen3-4b scheduler: admission, prefill/decode planning, effects
  -> pegainfer-qwen3-4b executor: prefix-cache match, KV state, prefill/decode forward
  -> pegainfer-kv-cache / kvbm-logical: logical block matching and residency
  -> TokenEvent stream back to frontend usage/output
```

## Requirements

* Extend the engine event contract so scheduled/prefill metadata can carry a per-request `cached_tokens` count.
* Preserve Qwen3 executor-computed `PrefillRequestResult.cached_tokens` through scheduler resolve/effects.
* Emit Qwen3 `TokenEvent::Scheduled` before any first token or terminal finished event for a request.
* Update vLLM frontend usage mapping to pass through cached-token counts from `TokenEvent::Scheduled`.
* Preserve existing behavior for non-Qwen3 engines by reporting `cached_tokens: 0` unless they later provide a real cache source.
* Add cheap local tests where possible for event/frontend pass-through and Qwen3 scheduler propagation.

## Final aligned implementation plan

1. Extend `TokenEvent::Scheduled` with `cached_tokens: usize`.
2. Update existing non-Qwen3 `Scheduled` senders/test helpers to pass `cached_tokens: 0`, preserving current behavior.
3. For Qwen3, preserve `queued_at_unix_s` from `GenerateRequest` and record the real `scheduled_at_unix_s` when a request is selected for prefill/unified execution.
4. Let Qwen3 executor continue owning prefix matching and producing `PrefillRequestResult.cached_tokens`.
5. Thread `PrefillRequestResult.cached_tokens` through Qwen3 `resolve_prefill_outputs` into pending effects.
6. Emit Qwen3 `TokenEvent::Scheduled { cached_tokens, ... }` after prefix matching/executor return but before any first `Token` or terminal `Finished` event.
7. Update vLLM frontend `PrefillStats` mapping to pass through `cached_tokens` from the event:
   * `num_prompt_tokens = prompt_tokens`
   * `num_computed_tokens = prompt_tokens.saturating_sub(cached_tokens)`
   * `num_cached_tokens = cached_tokens`
   * `num_local_cached_tokens = cached_tokens`
   * `num_external_cached_tokens = 0`
8. Do not make frontend infer, recompute, or guess cached-token counts.

## Timing decision

Other models currently send `TokenEvent::Scheduled` when the request enters scheduling/prefill-candidate flow. Qwen3 only knows the actual prefix-cache hit count after executor prefix matching.

Maintainer-aligned resolution: record the true `scheduled_at_unix_s` when Qwen3 selects the request for prefill/unified execution, then emit `TokenEvent::Scheduled` after executor return with that timestamp and the real `cached_tokens`. This is acceptable as long as the event is delivered before the first token or final response.

## Test requirements

* Add/adjust cheap local tests for event/frontend pass-through where possible.
* Cover both OpenAI response paths if feasible in this repo's existing tests:
  * non-streaming cold request: `usage.prompt_tokens_details.cached_tokens` is zero or absent.
  * non-streaming repeated warm request: `cached_tokens` is nonzero.
  * streaming cold request: usage reports zero or absent cached tokens.
  * streaming repeated warm request: usage reports nonzero cached tokens.
* GPU/model-weight validation is expected for the full Qwen3 cold/warm API behavior.
* Tests should verify frontend pass-through semantics rather than recomputing cache counts in the frontend.

## Acceptance Criteria

* [ ] API usage no longer hard-codes Qwen3 prefix-cache hits to zero when the scheduler reports real cached tokens.
* [ ] `TokenEvent::Scheduled` carries `cached_tokens` and all existing senders compile.
* [ ] Qwen3 scheduler emits `Scheduled` before request token/finish events.
* [ ] Qwen3 cached-token value is propagated from executor result into the frontend usage mapping.
* [ ] Non-Qwen3 behavior remains unchanged with `cached_tokens: 0`.
* [ ] Local tests/checks appropriate for touched crates pass or any GPU/model-weight-only gaps are documented.

## Out of Scope

* Re-implementing or revisiting #247; it is already merged separately.
* General operator hit-rate metrics beyond per-request usage propagation.
* Non-Qwen3 prefix-cache accounting.
* Tokenizer changes.
* Model/kernel behavior changes unrelated to cached-token reporting.

## References

* `issue-246-research.md` — archived research describing the original issue and likely impacted modules.
* Prior archived task: `.trellis/tasks/archive/2026-06/06-06-investigate-issue-246-as-first-project/`.
