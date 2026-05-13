//! Dispatch and combine plans.
//!
//! A "plan" carries the per-call routing information that varies between
//! invocations of the same backend: the routing indices, the per-rank /
//! per-expert token counts, and the dtypes of the payload. The backend
//! configuration (world size, device list, EP topology) lives on the
//! backend object itself and is established at construction time.

/// Dispatch plan: routing decisions for a single forward call.
///
/// Skeleton. Field shape is intentionally minimal; concrete fields will be
/// filled when wiring into PegaInfer's request scheduler. Adding fields is
/// not a breaking change as long as we keep this `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    /// Number of tokens fed into this dispatch (`bound_m` upstream).
    pub num_tokens: u32,
    /// Number of experts each token is routed to.
    pub num_experts_per_token: u32,
}

/// Combine plan: paired with a prior dispatch.
///
/// Skeleton. See [`DispatchPlan`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CombinePlan {
    /// Number of tokens that participated in the paired dispatch.
    pub num_tokens: u32,
    /// Whether the combine should accumulate into the output buffer.
    pub accumulate: bool,
}
