# Investigate issue 246 as first project

## Goal

Understand GitHub issue #246 well enough to choose a small, safe first-project implementation path in pegainfer.

## What I already know

* The user wants to review https://github.com/xiaguan/pegainfer/issues/246.
* The user wants to brainstorm whether it is suitable as a first project entry point.
* This should start with understanding the issue, affected code/docs, scope, risks, and likely MVP.
* Project context: pegainfer is a Rust + CUDA LLM inference engine with per-model crates, shared core runtime, and shared kernel crate boundaries.

## Assumptions (temporary)

* Issue #246 is public and can be read through GitHub tooling.
* The first-project goal favors a narrow, reviewable PR over a broad architectural change.
* Implementation should avoid GPU-heavy or model-weight-heavy validation unless the issue specifically requires it.

## Open Questions

* Confirm whether to implement #247 first and #246 second, or reverse the order.

## Requirements (evolving)

* Read and summarize issue #246.
* Identify likely impacted files/modules.
* Propose 2–3 feasible entry scopes with trade-offs.
* Converge on an MVP that is suitable for a first contribution.

## Acceptance Criteria (evolving)

* [ ] Issue problem statement is summarized accurately.
* [ ] Likely affected modules/files are identified.
* [ ] At least two implementation scopes are compared.
* [ ] MVP and out-of-scope work are explicit.
* [ ] Validation strategy is clear and realistic for this repository.

## Definition of Done (team quality bar)

* Tests added/updated where appropriate.
* Release-mode check/test command chosen for the touched package.
* Docs/notes updated if behavior changes.
* Rollout/rollback considered if risky.

## Out of Scope (explicit)

* Implementing code before the MVP is agreed.
* Broad refactors unrelated to issue #246/#247.
* Changing model/kernel behavior without a direct issue requirement.
* Combining both issues into one large PR unless explicitly chosen.

## Claim / Reply Strategy

* Reply on both issues to claim them, but be explicit that they will be handled as separate small PRs.
* For #247, say it looks like the best first PR because it is CPU-side kvbm-logical behavior and can be covered by standard cargo tests.
* For #246, say it will follow as cached-token propagation through scheduler/event/frontend usage, with final validation on a GPU/model-weight machine.
* Avoid promising exact timelines; state intended order and ask maintainers to redirect if they prefer the opposite order.

## Technical Approach for #246

Latest `main` still has the #246 gap:

* `pegainfer-engine/src/engine.rs` defines `TokenEvent::Scheduled` with only queue time, schedule time, and `prompt_tokens`.
* `pegainfer-vllm-frontend/src/lib.rs` builds `PrefillStats` from `TokenEvent::Scheduled` but hard-codes cached-token fields to zero.
* Qwen3 executor computes `PrefillRequestResult.cached_tokens` in `pegainfer-qwen3-4b/src/executor.rs`.
* Qwen3 scheduler currently does not emit `TokenEvent::Scheduled`, and `scheduler/resolve.rs` drops `PrefillRequestResult.cached_tokens` when converting prefill results into effects.

Execution chain for #246:

```text
HTTP /v1/completions
  -> pegainfer-server
  -> pegainfer-vllm-frontend: text/API bridge, tokenization/detokenization glue, response usage
  -> pegainfer-engine: GenerateRequest / TokenEvent contract
  -> pegainfer-qwen3-4b scheduler: admission, prefill/decode planning, effects
  -> pegainfer-qwen3-4b executor: prefix-cache match, KV state, prefill/decode forward
  -> pegainfer-kv-cache / kvbm-logical: logical block matching and residency
  -> pegainfer-kernels: CUDA/FlashInfer/cuBLAS/Triton kernels
  -> TokenEvent stream back to frontend usage/output
```

MVP design:

1. Add `cached_tokens: usize` to `TokenEvent::Scheduled`.
2. Update all non-Qwen3 `TokenEvent::Scheduled` senders/test helpers to pass `cached_tokens: 0`.
3. Preserve Qwen3 `PrefillRequestResult.cached_tokens` through `scheduler/resolve.rs` into pending effects.
4. Emit `TokenEvent::Scheduled` from Qwen3 `scheduler/effects.rs` before first token / finished events for pending prefill results, including `queued_at_unix_s`, `scheduled_at_unix_s`, `prompt_tokens`, and `cached_tokens`.
5. Update `pegainfer-vllm-frontend/src/lib.rs` so `PrefillStats` uses `cached_tokens`, sets local cached tokens to the same value, external cached tokens to zero, and computed tokens to `prompt_tokens.saturating_sub(cached_tokens)`.
6. Add cheap contract tests for event/frontend mapping and Qwen3 scheduler propagation; reserve full warm/cold API validation for GPU/model-weight machine.

Out of scope for #246 MVP:

* General operator hit-rate metrics beyond per-request usage propagation.
* Non-Qwen3 prefix-cache accounting; non-Qwen3 models should report `cached_tokens: 0` unless they gain a real cache source later.
* Tokenizer changes; frontend owns text/tokenization glue, while Qwen3 receives token ids.

### Open design question: `Scheduled` timing vs Qwen3 prefix matching

Other models currently send `TokenEvent::Scheduled` when the request enters the model engine's scheduling/prefill-candidate path, before prefill compute and before token output. The vLLM frontend uses that event as the early carrier for both queue/schedule timing and `PrefillStats`.

Qwen3 currently does not send `TokenEvent::Scheduled`. Its `cached_tokens` value is computed inside `Qwen3Executor::execute_prefill` / `execute_unified`, when `RequestKv::match_and_add_prefix` runs. That means the cache count is only available after executor prefill work starts, not at the exact moment the request is selected for prefill.

Potential MVP resolution: record the true `scheduled_at_unix_s` when Qwen3 selects the request for prefill/unified execution, then emit `TokenEvent::Scheduled { cached_tokens, ... }` after executor returns but before the first token/finished event. This preserves frontend ordering and timestamp correctness, but the event send time is later than the event's scheduled timestamp.

Maintainer decision: this direction is acceptable. `cached_tokens` should come from the engine/request prefill result, and the frontend should only pass it through into usage rather than recomputing or guessing. For Qwen3, it is fine if `TokenEvent::Scheduled` is emitted after prefix matching as long as it preserves the real scheduled timestamp and is delivered before the first token / final response. Tests should cover both streaming and non-streaming OpenAI responses: cold request reports zero/absent cached tokens; repeated warm request reports nonzero cached tokens; behavior should match vLLM-style usage reporting.

### Final aligned #246 implementation plan

* Extend `TokenEvent::Scheduled` with `cached_tokens: usize`.
* Update existing non-Qwen3 `Scheduled` senders to pass `cached_tokens: 0`, preserving current behavior.
* For Qwen3, preserve `queued_at_unix_s` from `GenerateRequest` and record the real `scheduled_at_unix_s` when a request is selected for prefill/unified execution.
* Let Qwen3 executor continue owning prefix matching and producing `PrefillRequestResult.cached_tokens`.
* Thread `PrefillRequestResult.cached_tokens` through Qwen3 `resolve_prefill_outputs` into pending effects.
* Emit Qwen3 `TokenEvent::Scheduled { cached_tokens, ... }` after prefix matching/executor return but before any first `Token` or terminal `Finished` event.
* Update vLLM frontend `PrefillStats` mapping to pass through `cached_tokens` from the event:
  * `num_prompt_tokens = prompt_tokens`
  * `num_computed_tokens = prompt_tokens.saturating_sub(cached_tokens)`
  * `num_cached_tokens = cached_tokens`
  * `num_local_cached_tokens = cached_tokens`
  * `num_external_cached_tokens = 0`
* Do not make frontend infer, recompute, or guess cached-token counts.

### Final aligned #246 test requirements

* Add/adjust cheap local tests for event/frontend pass-through where possible.
* Cover both OpenAI response paths:
  * non-streaming cold request: `usage.prompt_tokens_details.cached_tokens` is zero or absent.
  * non-streaming repeated warm request: `cached_tokens` is nonzero.
  * streaming cold request: usage reports zero/absent cached tokens.
  * streaming repeated warm request: usage reports nonzero cached tokens.
* GPU/model-weight validation is expected for the full Qwen3 cold/warm API behavior.
* Tests should verify frontend pass-through semantics rather than recomputing cache counts in the frontend.

## Technical Approach for #247

Implement #247 as a CPU-only behavioral test in `kvbm/kvbm-logical/src/integrations/scheduled.rs`, near the existing prefix matching tests.

### Test target

* Preferred test name: `test_prefix_cache_eviction_rematch_zero_then_recompute`.
* Use `SchedulableSequence`, not Qwen3 executor tests, because the issue targets CPU-side prefix matching/eviction and should run without GPU/model weights.
* Reuse existing local helpers in `scheduled.rs`: `TestMeta`, `create_test_manager`, `noop_delegate()`, `make_tokens()`, and the existing prefix-cache test style.

### Proposed test sequence

1. Create a small LRU-backed manager with 4 blocks and block size 4.
2. Register an original 12-token sequence as 3 complete blocks by scheduling/applying prefill.
3. Release the sequence so the prefix blocks become inactive and evictable.
4. Create a same-token warm sequence and call `match_and_add_prefix` before eviction.
   * Expected match is 2 blocks, not 3, because `SchedulableSequence` caps full matches so at least one input token remains to recompute.
   * Assert `prefill_position == 8`, `kv_position == 8`, and 2 assigned blocks.
5. Schedule/apply the remaining 4 tokens, assert prefill completes, then release again.
6. Force inactive eviction by calling `allocate_blocks_with_evictions(4)` and keep the returned pressure blocks alive.
   * Assert 4 blocks allocated and 3 prefix hashes evicted.
7. Create another same-token sequence and call `match_and_add_prefix` while pressure blocks are alive.
   * Expected match is 0, with no assigned blocks and positions still 0.
8. Drop the pressure blocks, schedule/apply the full sequence again, and assert recompute/re-register succeeds.
9. Release the recomputed sequence, then create one more same-token warm sequence and assert it again gets the maximum eligible prefix match.

### Assertions

* Before eviction: same prompt gets the maximum eligible prefix match under scheduled-layer cap.
* Eviction is actually forced: `evicted.len() == 3`.
* After eviction: same prompt gets zero prefix match, not stale reclaimed blocks.
* Recompute succeeds after pressure is released.
* After recompute and release, a later same-prefix request can match the re-registered prefix again.

### Validation

* Narrow iteration: `cargo test --release -p kvbm-logical --lib test_prefix_cache_eviction_rematch_zero_then_recompute`.
* Final local gate: `cargo test --release -p kvbm-logical --lib`.
* No GPU/model-weight validation is required for #247 MVP.

### Risks / Notes

* Be explicit in the test comments/assertion names that `SchedulableSequence` tests the Qwen3-facing scheduled-layer cap, so “full hit before eviction” means full eligible hit, not all physical blocks.
* Avoid sleeps/timing; eviction should be deterministic from block counts.
* Do not depend on inactive-pool internals beyond public allocation/match results.

## Technical Notes

* Task directory: `.trellis/tasks/06-06-investigate-issue-246-as-first-project/`.
* Relevant project rules loaded: `CLAUDE.md`, `docs/index.md`, `.trellis/spec/guides/index.md`, default backend spec index.

## Research References

* [`research/issue-246.md`](research/issue-246.md) — Issue #246 is a good-first-issue-shaped cross-crate plumbing task: executor computes cached tokens, but scheduler/frontend drop or zero them before API usage.
* [`research/issue-247.md`](research/issue-247.md) — Issue #247 is a separate CPU-side kvbm-logical eviction/rematch behavioral-test gap, not the same implementation path as #246.

## Research Notes

### Issue summary

* Issue #246: `qwen3: prefix-cache hits are invisible — cached_tokens never reaches the API`.
* Qwen3 prefix cache is enabled and executor already computes `PrefillRequestResult.cached_tokens`.
* The value is lost at the scheduler boundary and the vLLM frontend currently hard-codes cached-token usage stats to zero.
* Desired outcome: `usage.prompt_tokens_details.cached_tokens` reflects real per-request cache hits, and operators get some cache-effectiveness observability.

### Likely impacted areas

* Shared engine event contract: `TokenEvent::Scheduled` likely needs cached-token metadata.
* Qwen3 scheduler resolve/effects path must preserve `PrefillRequestResult.cached_tokens`.
* vLLM frontend bridge must map cached tokens into `PrefillStats` instead of hard-coding zero.
* Tests can cover executor behavior, scheduler/event propagation, frontend usage mapping, and optionally a full warm/cold API path with model weights.

### Relationship between #246 and #247

* Both issues are Qwen3 prefix-cache follow-ups and both are labeled `good first issue`.
* #246 is observability/plumbing across executor → scheduler → engine event → vLLM frontend API usage.
* #247 is correctness-test coverage for CPU-side block-manager eviction/rematch behavior.
* They can share a short design note and terminology, but implementation should be split unless the user explicitly wants one larger PR.

### Feasible first-project scopes

**Approach A: API usage plumbing only (#246)**

* Thread `cached_tokens` from Qwen3 executor result through scheduler effects/events into frontend `PrefillStats`.
* Pros: smallest scope that directly fixes the user-visible API metric.
* Cons: does not cover operator hit-rate logging from the issue's second desired outcome.

**Approach B: API usage plus minimal scheduler logging**

* Do Approach A and add low-noise scheduler-level logging/counters for prompt tokens, cached tokens, and hit requests.
* Pros: covers both API and operator visibility; still likely manageable.
* Cons: logging semantics need care to avoid noisy or misleading metrics.

**Approach C: CPU-friendly boundary tests first**

* Add fake-executor scheduler tests and frontend bridge tests proving cached-token propagation, then defer full warm/cold API integration.
* Pros: very safe first step, likely no GPU/model weights required.
* Cons: does not fully satisfy the issue acceptance wording by itself.
