# Research: GitHub issue 246 prefix-cache observability

- **Query**: Research GitHub issue https://github.com/xiaguan/pegainfer/issues/246 as a first project candidate, including issue metadata, problem statement, comments/maintainer hints, affected modules/files, suitability, MVP scopes, and validation strategy.
- **Scope**: mixed
- **Date**: 2026-06-06

## Findings

### Issue Metadata

| Field | Value |
|---|---|
| URL | https://github.com/xiaguan/pegainfer/issues/246 |
| Number | `#246` |
| Title | `qwen3: prefix-cache hits are invisible — cached_tokens never reaches the API` |
| State | `OPEN` |
| Author | `xiaguan` (`JinYan Su`) |
| Labels | `good first issue`, `qwen3` |
| Created | `2026-06-04T16:51:05Z` |
| Updated | `2026-06-04T16:51:05Z` |
| Comments | none returned by `gh issue view` |

### Problem Statement

Issue #246 says Qwen3-4B prefix caching is enabled by default after #216 and produces large warm-TTFT wins, but cache effectiveness is invisible to users, benchmarks, and operators.

The issue body states three current-state facts:

1. The executor computes a per-request cached-token count, but it is dropped at the scheduler boundary and never reaches the engine event stream.
2. The OpenAI-compatible frontend hard-codes `num_cached_tokens: 0` in usage at `pegainfer-vllm-frontend/src/lib.rs`.
3. There is no hit-rate logging, so production cache effectiveness is not observable.

Desired outcome:

- `usage.prompt_tokens_details.cached_tokens` reports the real per-request value.
- Engine-level hit/miss counters are logged or exposed for operators.

Acceptance boundary from the issue:

- An integration test asserts a warm repeat of the same prompt reports nonzero cached tokens while the cold run reports zero.
- Related surface: #78, streaming usage accounting, same area but independently fixable.

### Comments / Maintainer Hints

No issue comments were returned by `gh issue view 246 --json comments`; all maintainer hints are in the issue body and project docs.

Relevant documented hints:

- `docs/models/qwen3/roadmap.md:31` says `cached_tokens` is computed at `executor.rs:751` and dies at the scheduler boundary; frontend hardcodes `num_cached_tokens: 0`; expected work is to thread it through `TokenEvent::Scheduled` into usage and log hit rate.
- `docs/models/qwen3/prefix-cache.md:9` describes the executor-level matching path: `Qwen3Executor::execute_prefill` / `execute_unified` call `RequestKv::match_and_add_prefix` before scheduling; matched blocks advance `prefill_position`/`kv_position`; the uncached suffix is forwarded.
- `docs/models/qwen3/prefix-cache.md:10-11` says matching is full-block only with block size 16, and a full-block cap means even a fully-cached prompt still recomputes at least one token.
- `docs/models/qwen3/prefix-cache.md:54-57` points to existing direct executor tests that already assert cached-token counts and cached replay accuracy.

### Files Found

| File Path | Description |
|---|---|
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-engine/src/engine.rs:132-161` | Shared `TokenEvent` definition. `Scheduled` currently carries `queued_at_unix_s`, `scheduled_at_unix_s`, and `prompt_tokens`, but no `cached_tokens`. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs:30-43` | `PrefillStepItem` has `cached_tokens`, documented as leading prompt tokens whose KV came from the prefix cache. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs:148-154` | `build_prefill_request_results` copies `req.cached_tokens` into `PrefillRequestResult`. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs:368-374` | `PrefillRequestResult` includes public `cached_tokens: usize`. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs:740-753` | `execute_prefill` matches prefix cache and assigns `req.cached_tokens = rkv.match_and_add_prefix(&self.kv_mgr)?`. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs:843-854` | `execute_unified` does the same cached-token assignment for unified prefill+decode. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/plan.rs:193-211` | Scheduler builds `PrefillStepItem` from `PendingRequest` with `cached_tokens: 0`; executor later fills it. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/resolve.rs:30-89` | Scheduler resolves `PrefillRequestResult` into `PendingEffect`, but does not preserve `result.cached_tokens`. This is the boundary where executor data is lost. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/effects.rs:14-36` | `PendingEffect` variants do not include cached-token metadata, so `apply_effects` cannot emit it. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/effects.rs:162-216` | `apply_effects` emits first tokens and finished events for pending prefill results, but never emits `TokenEvent::Scheduled` for successful requests here. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler.rs:129-207` | Main scheduler loop drains, admits, executes, resolves, and applies effects. Candidate place for per-step/request counters if using scheduler-level logging. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler.rs:211-328` | LoRA-control scheduler loop mirrors the normal loop and would need consistent accounting if logging is added there. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:739-779` | vLLM bridge consumes `TokenEvent::Scheduled` and builds `PrefillStats`; it hard-codes `num_computed_tokens = prompt_tokens` and all cached-token fields to `0`. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:1417-1424` | Simulated frontend test helper sends `TokenEvent::Scheduled`; any `TokenEvent::Scheduled` shape change will affect tests/helpers. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:102-123` | Direct executor helper returns `PrefillRequestResult.cached_tokens`; existing behavioral test asserts executor-level cached counts. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:168-272` | Existing prefix-cache test checks cold zero, warm nonzero counts, mixed cold+warm batch counts, and unified-path counts directly against executor results. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/prefix-cache.md` | Living documentation for Qwen3 prefix cache behavior, block-size semantics, test commands, and pitfalls. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/roadmap.md:31` | Roadmap explicitly lists prefix-cache observability as a Now item and summarizes the likely implementation route. |

### Code Patterns

The cached-token value already exists in executor results but not in the public event stream:

```rust
// /Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs:148-154
outputs.push(PrefillRequestResult {
    request_id: req.request_id,
    first_token,
    first_token_logprob,
    prompt_logprobs,
    cached_tokens: req.cached_tokens,
});
```

The scheduler currently drops that value while resolving prefill results:

```rust
// /Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/resolve.rs:36-39
for (req, result) in pending.into_iter().zip(request_results) {
    debug_assert_eq!(req.request_id, result.request_id);
    let prompt_len = req.prompt_tokens.len();
```

`result.cached_tokens` is not copied into `PendingEffect` or `ActiveRequestState`.

The shared event contract currently has no cached-token field:

```rust
// /Users/jacky/projects/paga-projs/pegainfer/pegainfer-engine/src/engine.rs:132-137
pub enum TokenEvent {
    Scheduled {
        queued_at_unix_s: f64,
        scheduled_at_unix_s: f64,
        prompt_tokens: usize,
    },
```

The frontend has the hard-coded value called out by the issue:

```rust
// /Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs:772-778
first_token_prefill_stats = Some(PrefillStats {
    num_prompt_tokens: prompt_tokens as u32,
    num_computed_tokens: prompt_tokens as u32,
    num_cached_tokens: 0,
    num_local_cached_tokens: 0,
    num_external_cached_tokens: 0,
});
```

Existing direct executor tests already exercise the core cache behavior but not the API usage surface:

- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:184-188` asserts matching-disabled cached count is zero.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:195-205` asserts warm cached counts for repeated and extended prompts.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:224-225` asserts a mixed batch can contain both cold zero and warm nonzero cached counts.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:261-264` asserts unified-path warm prefill reports cached tokens.

The scheduler tests include a fake executor whose `PrefillRequestResult.cached_tokens` is always `0` at `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler.rs:749-771` and `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler.rs:814-850`. These fake paths may be useful for a cheap scheduler-boundary test, but they currently cannot demonstrate warm cache behavior unless extended.

### Related Specs / Docs

| Path | Relevance |
|---|---|
| `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/prefix-cache.md` | Source-of-truth documentation for Qwen3 prefix cache behavior and existing direct tests. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/roadmap.md` | Tracks prefix-cache observability as an open Qwen3 roadmap item with the same implementation hint as the issue. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/subsystems/frontend/simulated-inference-engine.md` | Frontend validation can be run through simulated infrastructure for CPU-only paths, but the specific warm-cache acceptance requires real Qwen3 cache behavior unless a test double is added. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/subsystems/scheduler/scheduler.md` | Background for scheduler architecture; relevant if adding scheduler-level counters/logging. |
| `/Users/jacky/projects/paga-projs/pegainfer/.trellis/spec/pegainfer-server/` | Package-specific spec directory exists for server/frontend-facing work; no specific prefix-cache contract found in the retrieved snippets. |

### External References

- GitHub issue #246: https://github.com/xiaguan/pegainfer/issues/246 — primary external source for metadata, acceptance boundary, and desired outcome.
- Related GitHub issue #78 (mentioned by #246, not fetched in this research): streaming usage accounting; adjacent surface but independently fixable according to #246.

### Likely Affected Repo Modules / Files

1. Shared engine contract:
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-engine/src/engine.rs`
   - Likely needs a cached-token field on an event, most directly `TokenEvent::Scheduled` per roadmap hint.

2. Qwen3 scheduler boundary:
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/resolve.rs`
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler/effects.rs`
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler.rs`
   - These files transform executor `PrefillRequestResult` into engine events and request state; `cached_tokens` is lost here today.

3. Qwen3 executor data source:
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs`
   - Already computes and returns `cached_tokens`; likely affected only as source/reference, not necessarily requiring changes.

4. Frontend usage bridge:
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs`
   - Converts `TokenEvent::Scheduled` into vLLM `PrefillStats`; currently hard-codes cached stats to zero.

5. Tests:
   - Existing executor-level: `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs`
   - Possible scheduler-boundary tests: inline tests in `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/scheduler.rs`
   - Possible frontend bridge tests: inline tests/helper paths in `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-vllm-frontend/src/lib.rs`
   - Possible end-to-end/API integration test would likely require server + Qwen3 weights, similar to existing GPU/model-weight tests.

### Suitability as a First Project

Why it looks suitable:

- The issue is labeled `good first issue` and `qwen3` by the repository.
- The data source is already implemented: executor `PrefillRequestResult.cached_tokens` is present and tested directly.
- The failure mode is localized and explicitly described: the value is dropped at scheduler boundary and frontend hard-codes zero.
- Existing tests in `pegainfer-qwen3-4b/tests/prefix_cache.rs` provide known prompt/cache behavior and expected cached-token counts.
- Most work is plumbing/accounting through Rust structs/enums rather than CUDA kernel changes.

Why it may be less ideal for a first project:

- It crosses crate boundaries: `pegainfer-engine`, `pegainfer-qwen3-4b`, and `pegainfer-vllm-frontend`.
- `TokenEvent` is a shared contract; changing its shape requires updating all pattern matches and test helpers.
- The acceptance test at the OpenAI usage/API layer likely needs GPU + Qwen3 model weights to exercise real prefix cache behavior.
- There are two scheduler loops (`scheduler_loop` and `scheduler_loop_with_lora_control`) and multiple completion paths (`Finish`, `EmitAndFinish`, `Promote`) that need consistent handling.
- Operator hit/miss counters are less clearly specified than per-request API usage; scope needs care to avoid growing beyond the good-first-issue core.

### Possible MVP Scopes and Trade-offs

#### MVP 1: Per-request cached tokens through API usage only

Thread `cached_tokens` from `PrefillRequestResult` through scheduler effects/event stream into `TokenEvent::Scheduled`, then have the vLLM frontend set `PrefillStats.num_cached_tokens` and `num_local_cached_tokens` from it and `num_computed_tokens = prompt_tokens - cached_tokens`.

Trade-offs:

- Best fit for the explicit acceptance boundary around `usage.prompt_tokens_details.cached_tokens`.
- Requires changing the shared `TokenEvent` shape and updating all `Scheduled` senders/matches.
- Does not fully satisfy the operator-level hit-rate logging/exposure desired outcome unless paired with minimal logs.

#### MVP 2: API usage plus minimal scheduler logging

Do MVP 1 and add scheduler-level log lines/counters when prefill results are resolved, e.g. total prompt tokens, cached tokens, cache-hit request count for each step or cumulative scheduler lifetime.

Trade-offs:

- Covers both user/API visibility and the issue's operator-observability direction.
- Still relatively small because the scheduler already sees every `PrefillRequestResult` in `resolve_prefill_outputs`.
- Logging semantics need to be chosen: per-step logs may be noisy, while cumulative counters need storage in scheduler loop state or a small accounting object.

#### MVP 3: Test-first boundary plumbing with fake executor, defer full API integration

Extend scheduler fake executor/tests to return configurable `cached_tokens`, assert the scheduler emits it on `TokenEvent::Scheduled`, and add/update frontend bridge tests to assert `PrefillStats` uses the event value.

Trade-offs:

- CPU-only and newcomer-friendly; avoids GPU/model-weight dependency while proving the dropped-boundary bug.
- Does not by itself satisfy the issue's explicit integration-test wording: warm repeat prompt cold=0/warm>0 at API usage layer.
- Useful as an intermediate if the full Qwen3 integration test is expensive or flaky.

### Validation Strategy

1. Static/build validation:
   - Run `cargo check --release --workspace` or narrower package checks after changing the shared `TokenEvent` shape.
   - Because project guidance says CUDA/debug builds are slow and release should be used, avoid debug `cargo test` for GPU paths.

2. Existing Qwen3 prefix-cache behavior gate:
   - Run the existing direct executor test with weights:
     `PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test prefix_cache`
   - This confirms executor cached-token counts still behave as documented.

3. Scheduler-boundary unit/integration validation:
   - Add or update a scheduler test using a fake executor that returns nonzero `PrefillRequestResult.cached_tokens` and assert the receiver observes a `TokenEvent::Scheduled { cached_tokens: ... }` or equivalent event metadata before token output.
   - This isolates the exact boundary called out by the issue.

4. Frontend usage validation:
   - Add/update frontend bridge tests so a `TokenEvent::Scheduled` with `prompt_tokens = N` and `cached_tokens = C` produces `PrefillStats` with `num_prompt_tokens = N`, `num_computed_tokens = N - C`, `num_cached_tokens = C`, and `num_local_cached_tokens = C`.

5. Acceptance-level warm/cold validation:
   - With Qwen3 weights and server/frontend path, send a cold request and then a warm repeat of the same prompt.
   - Assert cold `usage.prompt_tokens_details.cached_tokens == 0` and warm repeat is nonzero.
   - Use a prompt long enough to include at least one full 16-token block; existing tests use examples like 50 tokens where 3 full blocks can hit.
   - Drain the stream to completion between requests, per `docs/models/qwen3/prefix-cache.md:48-52`, so zombie decode does not pollute timing or cache observation.

## Caveats / Not Found

- No issue comments were found; research relies on the issue body plus local docs/code.
- This research did not fetch issue #78; it is only noted because #246 names it as adjacent but independently fixable.
- `gh issue view` succeeded, so no web fallback was needed.
- I did not modify code or run tests; validation commands above are strategy only.
- The exact OpenAI response type that materializes `usage.prompt_tokens_details.cached_tokens` is inside the vLLM frontend/protocol layer; this research traced pegainfer's bridge to `PrefillStats`, not the downstream vLLM serialization internals.
