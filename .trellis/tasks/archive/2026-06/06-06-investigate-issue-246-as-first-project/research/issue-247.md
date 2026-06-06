# Research: GitHub issue 247 prefix-cache eviction behavioral test

- **Query**: Research GitHub issue https://github.com/xiaguan/pegainfer/issues/247 for task directory /Users/jacky/projects/paga-projs/pegainfer/.trellis/tasks/06-06-investigate-issue-246-as-first-project. Include issue metadata, problem statement, maintainer hints, affected files, relationship to #246, whether to design/implement together or split, MVP scope options, and validation strategy for local CPU-only plus separate GPU/model-weight machine.
- **Scope**: mixed
- **Date**: 2026-06-06

## Findings

### Issue Metadata

| Field | Value |
|---|---|
| URL | https://github.com/xiaguan/pegainfer/issues/247 |
| Number | `#247` |
| Title | `qwen3: prefix-cache eviction path has no behavioral test` |
| State | `OPEN` |
| Author | `xiaguan` (`JinYan Su`) |
| Labels | `good first issue`, `qwen3` |
| Created | `2026-06-04T16:51:08Z` |
| Updated | `2026-06-04T16:51:08Z` |
| Comments | none returned by `gh issue view` |

### Problem Statement

Issue #247 says Qwen3-4B prefix caching can evict inactive blocks under pool pressure, and correctness after eviction depends on the radix/block-matching layer returning only blocks that are actually resident.

The issue body states:

1. Prefix cache was introduced in #216 and evicts blocks under pool pressure.
2. Current accuracy gates replay cache-hit passes, but do not exercise the sequence `register prefix → release prefix → force eviction → match again`.
3. A partial-eviction bug could match blocks that were reclaimed, causing silent production output corruption rather than a test failure.
4. The relevant block-matching layer is CPU-side logic and should not require GPU.

Desired outcome from the issue:

- A behavioral test for the evict-then-rematch cycle:
  - full hit before eviction,
  - truncated or zero match after eviction,
  - correct recompute on the next request.

Acceptance boundary from the issue:

- The test runs in the standard `cargo test` suite without GPU or model weights.

### Comments / Maintainer Hints

No issue comments were returned by `gh issue view 247 --json comments`; all maintainer hints are in the issue body and local docs.

Relevant local hints:

- `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/roadmap.md:38` lists “Eviction behavioral test” as an open Qwen3 item: “Evict-then-remiss is never exercised: register a prefix, release it, pressure the pool until eviction, assert truncated/zero match + correct recompute. kvbm-logical layer needs no GPU.”
- `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/prefix-cache.md:9-14` describes the current Qwen3 prefix cache wiring: executor prefill/unified paths call `RequestKv::match_and_add_prefix`; matching is full-block only; matched blocks are held as strong `ImmutableBlock` references for request lifetime, so inactive-pool LRU cannot reclaim them mid-request.
- `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/prefix-cache.md:54-57` says existing `pegainfer-qwen3-4b/tests/prefix_cache.rs` pins cached-token counts and warm-vs-cold logit behavior, but that file requires CUDA GPU and Qwen3 weights.

### Files Found

| File Path | Description |
|---|---|
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-kv-cache/src/manager.rs:13-126` | Pegainfer wrapper around `kvbm_logical::BlockManager`; constructs an LRU-backed manager, reserves padding block, and creates `RequestKv` with block-size and LoRA salt. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-kv-cache/src/manager.rs:136-152` | `RequestKv::match_and_add_prefix` delegates to `SchedulableSequence::match_and_add_prefix` and returns matched token count. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-kv-cache/src/manager.rs:156-209` | `RequestKv` prefill/decode schedule/apply/release lifecycle; `release()` drops assignments so registered blocks can move to inactive/evictable state. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/integrations/scheduled.rs:416-447` | `SchedulableSequence::match_and_add_prefix` caps matches so at least one input token remains uncached, advances `prefill_position` and `kv_position`, and emits a `PrefixMatched` event. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/integrations/scheduled.rs:453-568` | CPU-side two-phase prefill schedule/apply path that allocates blocks and registers completed prefix blocks. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/integrations/request.rs:183-260` | Low-level prefix matching: `RequestSequence::match_and_add_prefix` calls `match_prefix`, truncates to `max_blocks`, validates sequence hashes, and adds matched `ImmutableBlock`s. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/integrations/request.rs:361-417` | `release()` clears assignments; `reacquire()` rematches available prefix blocks and allocates/registers the remainder. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/manager/mod.rs:58-74` | `BlockManager::allocate_blocks`/`allocate_blocks_with_evictions` allocate from reset and evict inactive blocks when needed. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/manager/mod.rs:110-152` | `BlockManager::match_blocks` does linear prefix match against active/inactive pools and stops at the first miss. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/pools/store.rs:4-24` | `BlockStore` documentation: unified single-mutex bookkeeping for reset, active, inactive pools; active lookup, slot transitions, and resurrection happen under one lock. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/pools/store.rs:103-151` | Slot states include `Reset`, `Mutable`, `Staged`, `Primary`, `Duplicate`, and `Inactive`; inactive blocks are idle, evictable, registered, and indexed by sequence hash. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/pools/inactive/backends/lru_backend.rs:29-86` | LRU inactive index: prefix find stops on first miss; allocation evicts least-recently-used inactive blocks. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/manager/tests.rs` | Existing CPU-side block manager tests cover allocation, inactive pool, reset-on-release, resurrection, and eviction-back-to-mutable details. Search did not find the exact issue #247 evict-then-rematch sequence. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/integrations/request.rs:675-708` | Existing request integration tests include full and partial prefix-cache hits. |
| `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:168-272` | Existing GPU/model-weight behavioral test asserts warm cached-token counts, extension hits, full-block cap, mixed batch, unified path, and logit closeness, but not eviction after release/pressure. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/prefix-cache.md` | Living Qwen3 prefix-cache doc with behavior, pitfalls, and existing test commands. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/roadmap.md:31-38` | Qwen3 roadmap lists #246-style observability and #247-style eviction behavioral test as separate open items. |
| `/Users/jacky/projects/paga-projs/pegainfer/Cargo.toml:37-49` | `kvbm/kvbm-logical` and `pegainfer-kv-cache` are workspace members, so CPU-only `cargo test -p kvbm-logical --lib` or `cargo test -p pegainfer-kv-cache` style validation should be available without model weights. |

### Code Patterns

The Qwen3 executor uses `pegainfer-kv-cache::RequestKv`, but the issue’s target behavior can be exercised below the executor because matching and eviction are CPU-side block-lifecycle logic.

Qwen3-facing wrapper:

```rust
// /Users/jacky/projects/paga-projs/pegainfer/pegainfer-kv-cache/src/manager.rs:146-152
pub fn match_and_add_prefix(&mut self, manager: &KvCacheManager) -> anyhow::Result<usize> {
    let blocks = self
        .seq
        .match_and_add_prefix(&manager.block_manager)
        .map_err(|e| anyhow::anyhow!("match_and_add_prefix: {e}"))?;
    Ok(blocks * self.seq.block_size())
}
```

The cap that keeps at least one input token uncached is in `SchedulableSequence`:

```rust
// /Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/integrations/scheduled.rs:430-439
let bs = self.inner.block_size();
let max_blocks = self.inner.num_input_tokens().saturating_sub(1) / bs;
let count = self
    .inner
    .match_and_add_prefix(manager, max_blocks)
    .unwrap_or_else(|_| panic!("prefix match should not produce duplicates"));

if count > 0 {
    self.prefill_position += count * self.inner.block_size();
    self.kv_position = self.prefill_position;
}
```

The matching contract is linear prefix matching, stopping at the first miss:

```rust
// /Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/manager/mod.rs:119-129
pub fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
    self.metrics
        .inc_match_hashes_requested(seq_hash.len() as u64);

    if seq_hash.is_empty() {
        self.metrics.inc_match_blocks_returned(0);
        return Vec::new();
    }

    let inners = self.store.match_prefix_locked_batch(seq_hash);
```

The inactive LRU backend also stops prefix lookup at the first miss, which is directly relevant to “partial eviction truncates matches”:

```rust
// /Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/pools/inactive/backends/lru_backend.rs:29-43
impl InactiveIndex for LruBackend {
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some(block_id) = self.cache.pop(hash) {
                matches.push((*hash, block_id));
            } else {
                break;
            }
        }
        matches
    }
```

Existing Qwen3 prefix-cache tests cover cache hits but not eviction:

- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:184-205` asserts cold zero and warm nonzero counts.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:207-213` asserts full-block cap behavior.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:215-245` asserts mixed cold/warm batch behavior.
- `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs:247-269` asserts unified prefill+decode uses the same match path.

### Relationship to Issue #246

Issue #246 metadata from `gh issue view`:

| Field | Value |
|---|---|
| URL | https://github.com/xiaguan/pegainfer/issues/246 |
| Title | `qwen3: prefix-cache hits are invisible — cached_tokens never reaches the API` |
| State | `OPEN` |
| Author | `xiaguan` (`JinYan Su`) |
| Labels | `good first issue`, `qwen3` |
| Comments | none returned by `gh issue view` |

Relationship:

- Both issues are Qwen3 prefix-cache follow-ups after #216 and both are labeled `good first issue` + `qwen3`.
- #246 is observability/plumbing: cached-token counts exist in executor results but are dropped before engine/frontend usage reporting.
- #247 is correctness-test coverage: block matching after inactive-pool eviction must not return reclaimed blocks.
- #246 crosses engine/frontend/API layers; #247 can be implemented as a CPU-only logical block/cache test.
- The issues share terminology (`cached_tokens`, prefix hit/miss), but they target different failure classes.

Whether to design/implement together or split:

- Best split for implementation. #247’s acceptance boundary explicitly says no GPU/model weights and points to CPU-side block matching; #246’s acceptance boundary needs API usage accounting and likely a GPU/model-weight warm-repeat integration test for full confidence.
- Light design coordination is useful only for naming and documentation: #247 can establish exact hit/truncated/zero behavior; #246 can later expose the observed per-request cached-token count to users/operators.
- Combining both in one implementation would cross more crates (`kvbm-logical`/`pegainfer-kv-cache` plus `pegainfer-engine`/scheduler/frontend) and dilute the first-project scope.

### Likely Affected Repo Modules / Files

1. Primary CPU-side test target:
   - `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/integrations/request.rs`
   - `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/integrations/scheduled.rs`
   - `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/src/manager/tests.rs`

2. Pegainfer wrapper-level possible test target:
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-kv-cache/src/manager.rs`

3. Reference / existing GPU behavior, likely not the main target for #247:
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/src/executor.rs`
   - `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs`

4. Docs that mention the gap:
   - `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/roadmap.md:38`
   - `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/prefix-cache.md`

### Related Specs / Docs

| Path | Relevance |
|---|---|
| `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/prefix-cache.md` | Current behavior and existing prefix-cache tests; confirms matching/full-block/cap/lifetime semantics. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/models/qwen3/roadmap.md` | Explicitly lists both prefix-cache observability and eviction behavioral test as separate Qwen3 roadmap items. |
| `/Users/jacky/projects/paga-projs/pegainfer/docs/subsystems/runtime/kv-cache-design.md` | General KV cache design background: reset/active/inactive pools, inactive eviction policy, content-addressed blocks. |
| `/Users/jacky/projects/paga-projs/pegainfer/.trellis/spec/guides/cross-layer-thinking-guide.md` | Relevant mainly for #246; #247 is intentionally narrower because it does not need cross-layer API plumbing. |
| `/Users/jacky/projects/paga-projs/pegainfer/kvbm/kvbm-logical/CLAUDE.md` | Local crate guidance: `kvbm-logical` block lifecycle, test utilities, and CPU test commands. |

### External References

- GitHub issue #247: https://github.com/xiaguan/pegainfer/issues/247 — primary source for metadata, problem statement, and acceptance boundary.
- GitHub issue #246: https://github.com/xiaguan/pegainfer/issues/246 — related prefix-cache observability issue, fetched for relationship analysis.

### MVP Scope Options and Trade-offs

#### MVP 1: Pure `kvbm-logical` integration test for evict-then-rematch

Add a CPU-only test around `SchedulableSequence` or `RequestSequence` that:

1. Builds a small LRU-backed `BlockManager` with enough blocks for one registered prefix and a little pressure.
2. Registers a multi-block prefix by scheduling/applying prefill or using existing request test utilities.
3. Releases the sequence so blocks enter inactive state.
4. Confirms a fresh same-prefix request gets a full hit before eviction, subject to the full-block cap in `SchedulableSequence` if using that layer.
5. Releases again, allocates/registers other blocks until inactive eviction occurs.
6. Attempts the original prefix again and asserts the match is truncated or zero, not stale/full.
7. Schedules/applies the remainder to prove recompute/re-register succeeds.

Trade-offs:

- Best fit for issue acceptance: no GPU, no model weights, standard `cargo test`.
- Tests the exact CPU-side matching/eviction layer named by the issue.
- Slightly less Qwen3-shaped unless token counts/block size mirror Qwen3-style full-block behavior.

#### MVP 2: `pegainfer-kv-cache` wrapper-level CPU test

Add a test around `pegainfer-kv-cache::KvCacheManager`/`RequestKv` APIs, if practical without constructing CUDA `KvBuffer`; otherwise this may not be CPU-only because `KvCacheManager::new` currently allocates a GPU `KvBuffer`.

Trade-offs:

- Closer to the Qwen3 executor’s actual API (`RequestKv::match_and_add_prefix`, `schedule_prefill`, `apply_prefill`, `release`).
- May violate #247’s no-GPU acceptance if `KvCacheManager::new` requires `CudaStream`/`KvBuffer` allocation.
- If a CPU-only manager wrapper/test seam already exists or is added, it could become the most representative non-GPU test.

#### MVP 3: Extend existing Qwen3 GPU prefix-cache test with eviction scenario plus a smaller CPU unit

Add a GPU/model-weight scenario to `/Users/jacky/projects/paga-projs/pegainfer/pegainfer-qwen3-4b/tests/prefix_cache.rs` that pressures the real executor’s KV manager and checks recompute/logit closeness after eviction, while also adding a CPU logical test for standard suite acceptance.

Trade-offs:

- Highest end-to-end confidence because it exercises real Qwen3 executor state.
- Not sufficient alone: issue explicitly asks for standard `cargo test` without GPU/model weights.
- More expensive and harder as a first project; useful as a follow-up validation rather than MVP.

### Validation Strategy

Assumption: local machine has no GPU; separate dev machine can run GPU/model-weight tests.

Local CPU-only validation:

1. Run the targeted logical test package:
   - `cargo test --release -p kvbm-logical --lib`
   - Or narrower while iterating: `cargo test --release -p kvbm-logical --lib <test_name>`
2. If the test is placed in/near `pegainfer-kv-cache`, run the narrow package test that does not instantiate CUDA. If no CPU-only seam exists there, prefer `kvbm-logical` for the acceptance test.
3. Run `cargo test --release --workspace --lib` only if local toolchain/build prerequisites are available; project guidance says release builds should be used for CUDA/GPU code, and the issue’s intended test should not need GPU.
4. For deterministic assertions, use small block sizes and block counts so eviction order is forced by the LRU inactive backend. Avoid timing/sleeps; force pressure by allocations.

Separate GPU/model-weight validation:

1. Existing Qwen3 prefix-cache behavior gate:
   - `PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test prefix_cache`
2. Existing HF golden gate if broader confidence is needed:
   - `PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test hf_golden_gate`
3. If an optional Qwen3 eviction scenario is added, use prompts with several full 16-token blocks and enough unique prompts/requests to pressure the executor KV pool; assert cached-token counts truncate/zero after eviction and warm-vs-cold logits remain within existing bf16 tolerances.
4. If issue #246 is implemented separately, run API/frontend warm-repeat validation there; it is not necessary for #247’s CPU-only acceptance.

### Caveats / Not Found

- No issue comments were found; research relies on issue bodies plus local docs/code.
- `gh issue view` succeeded, so no web fallback was needed.
- I did not run tests or modify code; validation above is a strategy only.
- Search found many inactive-pool and resurrection tests in `kvbm-logical`, but not the exact #247 sequence: register prefix, release it, force eviction, rematch same prefix, and assert truncated/zero match plus recompute.
- `pegainfer-kv-cache::KvCacheManager` currently owns a `KvBuffer` allocated from `CudaStream`, so a CPU-only acceptance test is more likely to belong directly in `kvbm-logical` unless a non-GPU test seam already exists.
