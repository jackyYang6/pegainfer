use super::engine::DpRankState;

pub(super) struct DpLoadBalancer;

impl DpLoadBalancer {
    pub(super) fn new(_dp_world: usize) -> Self {
        Self
    }

    pub(super) fn pick_rank(&self, ranks: &[DpRankState]) -> Option<usize> {
        ranks
            .iter()
            .enumerate()
            .filter(|(_, r)| r.has_free_slot())
            .max_by_key(|(_, r)| r.free_slot_count())
            .map(|(i, _)| i)
    }
}
