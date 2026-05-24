//! Model-agnostic parallel topology types.

/// Pure parallel topology. No model-specific fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParallelConfig {
    pub tp_world: usize,
    pub dp_world: usize,
    pub ep_world: usize,
}

impl ParallelConfig {
    #[must_use]
    pub fn new(tp_world: usize, dp_world: usize) -> Self {
        assert!(tp_world > 0 && dp_world > 0);
        Self {
            tp_world,
            dp_world,
            ep_world: tp_world * dp_world,
        }
    }

    #[must_use]
    pub fn coord(&self, global_rank: usize) -> RankCoord {
        assert!(global_rank < self.ep_world);
        RankCoord {
            global_rank,
            tp_rank: global_rank % self.tp_world,
            dp_rank: global_rank / self.tp_world,
            ep_rank: global_rank,
        }
    }

    #[must_use]
    pub fn tp_group(&self, dp_rank: usize) -> std::ops::Range<usize> {
        let start = dp_rank * self.tp_world;
        start..start + self.tp_world
    }
}

/// A rank's coordinate in the TP×DP×EP grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankCoord {
    pub global_rank: usize,
    pub tp_rank: usize,
    pub dp_rank: usize,
    pub ep_rank: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp8_dp1() {
        let cfg = ParallelConfig::new(8, 1);
        assert_eq!(cfg.ep_world, 8);
        let c = cfg.coord(3);
        assert_eq!(c.tp_rank, 3);
        assert_eq!(c.dp_rank, 0);
        assert_eq!(c.ep_rank, 3);
    }

    #[test]
    fn tp1_dp8() {
        let cfg = ParallelConfig::new(1, 8);
        assert_eq!(cfg.ep_world, 8);
        let c = cfg.coord(3);
        assert_eq!(c.tp_rank, 0);
        assert_eq!(c.dp_rank, 3);
        assert_eq!(c.ep_rank, 3);
    }

    #[test]
    fn tp2_dp4() {
        let cfg = ParallelConfig::new(2, 4);
        assert_eq!(cfg.ep_world, 8);
        let c = cfg.coord(5);
        assert_eq!(c.tp_rank, 1);
        assert_eq!(c.dp_rank, 2);
        assert_eq!(c.ep_rank, 5);
        assert_eq!(cfg.tp_group(2), 4..6);
    }
}
