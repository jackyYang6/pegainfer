use std::sync::mpsc as std_mpsc;

use anyhow::Result;
use pegainfer_core::engine::{FinishReason, GenerateRequest, TokenEvent};
use tokio::sync::mpsc;

use super::executor::ForwardExecutor;
use super::worker::KimiOneTokenForwardReport;

const MAX_BATCH_PER_DP: usize = 4;

/// Coordinated DP engine: one coordinator thread drives all DP ranks in
/// lock-step. Every decode step, ALL ranks execute forward simultaneously
/// (active ranks with real tokens, idle ranks with padding). This satisfies
/// the PPLX EP contract that requires all ranks to participate in every
/// MoE layer's dispatch/combine collective.
pub(super) struct DpCoordinator {
    dp_world: usize,
    ranks: Vec<DpRankState>,
    executors: Vec<Box<dyn ForwardExecutor + Send>>,
    step_txs: Vec<std_mpsc::SyncSender<StepCommand>>,
    result_rxs: Vec<std_mpsc::Receiver<StepResult>>,
}

pub(super) struct DpRankState {
    slots: Vec<Option<RequestState>>,
}

struct RequestState {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    prompt_len: usize,
    completion_tokens: usize,
    max_tokens: usize,
    last_token: u32,
}

enum StepCommand {
    Decode {
        token_ids: Vec<u32>,
        positions: Vec<usize>,
        slots: Vec<usize>,
        batch_size: usize,
    },
    Prefill {
        input_ids: Vec<u32>,
        slot: usize,
        decode_batch_size: usize,
        ep_max_seq_len: usize,
    },
    Shutdown,
}

enum StepResult {
    Decode(Result<Vec<KimiOneTokenForwardReport>>),
    Prefill(Result<KimiOneTokenForwardReport>),
}

impl DpCoordinator {
    pub(super) fn new(executors: Vec<Box<dyn ForwardExecutor + Send>>) -> Self {
        let dp_world = executors.len();
        let mut ranks = Vec::with_capacity(dp_world);
        for _ in 0..dp_world {
            ranks.push(DpRankState {
                slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
            });
        }

        Self {
            dp_world,
            ranks,
            executors,
            step_txs: Vec::new(),
            result_rxs: Vec::new(),
        }
    }

    /// Spawn per-rank forward threads and run the coordinated decode loop.
    /// This consumes self and blocks until shutdown.
    pub(super) fn run(
        mut self,
        mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
        lb: super::load_balancer::DpLoadBalancer,
    ) {
        let mut step_txs = Vec::with_capacity(self.dp_world);
        let mut result_rxs = Vec::with_capacity(self.dp_world);
        let mut handles = Vec::with_capacity(self.dp_world);

        for (dp_rank, executor) in self.executors.drain(..).enumerate() {
            let (cmd_tx, cmd_rx) = std_mpsc::sync_channel::<StepCommand>(1);
            let (res_tx, res_rx) = std_mpsc::sync_channel::<StepResult>(1);
            step_txs.push(cmd_tx);
            result_rxs.push(res_rx);

            let handle = std::thread::Builder::new()
                .name(format!("kimi-k2-dp-fwd-{dp_rank}"))
                .spawn(move || {
                    rank_forward_loop(executor, cmd_rx, res_tx);
                })
                .expect("failed to spawn DP rank forward thread");
            handles.push(handle);
        }

        self.step_txs = step_txs;
        self.result_rxs = result_rxs;

        let mut pending_reqs: Vec<GenerateRequest> = Vec::new();

        loop {
            // 1. Drain new requests from submit channel
            if self.global_active_count() == 0 && pending_reqs.is_empty() {
                match submit_rx.blocking_recv() {
                    Some(req) => pending_reqs.push(req),
                    None => break,
                }
            }
            while let Ok(req) = submit_rx.try_recv() {
                pending_reqs.push(req);
            }

            // 2. Admit pending requests to DP ranks via load balancer
            let mut still_pending = Vec::new();
            for req in pending_reqs.drain(..) {
                let dp_rank = lb.pick_rank(&self.ranks);
                match dp_rank {
                    Some(rank) => self.admit_request(rank, req),
                    None => still_pending.push(req),
                }
            }
            pending_reqs = still_pending;

            // 3. Run one synchronized step across ALL ranks
            if self.global_active_count() > 0 {
                self.synchronized_decode_step();
            }
        }

        // Shutdown all forward threads
        for tx in &self.step_txs {
            let _ = tx.send(StepCommand::Shutdown);
        }
        for handle in handles {
            let _ = handle.join();
        }
    }

    fn global_active_count(&self) -> usize {
        self.ranks.iter().map(|r| r.active_count()).sum()
    }

    fn admit_request(&mut self, dp_rank: usize, req: GenerateRequest) {
        let slot = match self.ranks[dp_rank].find_free_slot() {
            Some(s) => s,
            None => {
                let _ = req.token_tx.send(TokenEvent::Rejected {
                    message: format!("DP rank {dp_rank} has no free slots"),
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                return;
            }
        };

        let scheduled_at = unix_now_s();
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s: req.queued_at_unix_s.unwrap_or(scheduled_at),
            scheduled_at_unix_s: scheduled_at,
            prompt_tokens: req.prompt_tokens.len(),
        });

        if req.max_tokens == 0 {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
            return;
        }
        if req.prompt_tokens.is_empty() {
            let _ = req.token_tx.send(TokenEvent::Rejected {
                message: "Kimi-K2 forward requires at least one prompt token".into(),
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            return;
        }

        // Prefill: all ranks run prefill in lock-step so PPLX collectives
        // align. Owning rank processes real tokens; padding ranks process a
        // single dummy token (output discarded).
        self.synchronized_prefill(dp_rank, slot, &req);

        let prompt_len = req.prompt_tokens.len();
        let last_token = match &self.result_rxs[dp_rank].recv() {
            Ok(StepResult::Prefill(Ok(report))) => {
                let token_id = report.local_next_token_global_id;
                if req
                    .token_tx
                    .send(TokenEvent::Token {
                        id: token_id,
                        logprob: None,
                    })
                    .is_err()
                {
                    return;
                }
                token_id
            }
            Ok(StepResult::Prefill(Err(err))) => {
                eprintln!("kimi-k2: DP rank {dp_rank} prefill failed: {err:#}");
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: format!("Kimi-K2 DP rank {dp_rank} prefill failed: {err:#}"),
                    prompt_tokens: prompt_len,
                    completion_tokens: 0,
                });
                return;
            }
            _ => return,
        };

        // Drain padding results from other ranks
        for r in 0..self.dp_world {
            if r != dp_rank {
                let _ = self.result_rxs[r].recv();
            }
        }

        let completion_tokens = 1;
        if completion_tokens >= req.max_tokens {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens,
            });
            return;
        }

        self.ranks[dp_rank].slots[slot] = Some(RequestState {
            token_tx: req.token_tx,
            prompt_len,
            completion_tokens,
            max_tokens: req.max_tokens,
            last_token,
        });
    }

    fn synchronized_prefill(&self, owning_rank: usize, slot: usize, req: &GenerateRequest) {
        let ep_max_seq_len = req.prompt_tokens.len();
        for dp_rank in 0..self.dp_world {
            let cmd = if dp_rank == owning_rank {
                StepCommand::Prefill {
                    input_ids: req.prompt_tokens.clone(),
                    slot,
                    decode_batch_size: 1,
                    ep_max_seq_len,
                }
            } else {
                // All ranks run prefill so they traverse layers at the same
                // pace, making exactly 1 PPLX dispatch/combine per MoE layer.
                StepCommand::Prefill {
                    input_ids: vec![0],
                    slot: 0,
                    decode_batch_size: 1,
                    ep_max_seq_len,
                }
            };
            let _ = self.step_txs[dp_rank].send(cmd);
        }
    }

    fn synchronized_decode_step(&mut self) {
        // Build per-rank decode commands
        for dp_rank in 0..self.dp_world {
            let cmd = self.ranks[dp_rank].build_decode_command();
            let _ = self.step_txs[dp_rank].send(cmd);
        }

        // Collect results from all ranks
        for dp_rank in 0..self.dp_world {
            let result = match self.result_rxs[dp_rank].recv() {
                Ok(StepResult::Decode(Ok(reports))) => reports,
                Ok(StepResult::Decode(Err(err))) => {
                    eprintln!("kimi-k2: DP rank {dp_rank} decode failed: {err:#}");
                    self.ranks[dp_rank].fail_all_active(&err);
                    continue;
                }
                _ => continue,
            };

            self.ranks[dp_rank].process_decode_results(result);
        }
    }
}

impl DpRankState {
    pub(super) fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub(super) fn free_slot_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_none()).count()
    }

    pub(super) fn has_free_slot(&self) -> bool {
        self.slots.iter().any(|s| s.is_none())
    }

    fn find_free_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.is_none())
    }

    fn build_decode_command(&self) -> StepCommand {
        let active: Vec<(usize, &RequestState)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|r| (i, r)))
            .collect();

        if active.is_empty() {
            return build_padding_decode_command();
        }

        let batch_size = active.len();
        let token_ids: Vec<u32> = active.iter().map(|(_, r)| r.last_token).collect();
        let positions: Vec<usize> = active
            .iter()
            .map(|(_, r)| r.prompt_len + r.completion_tokens - 1)
            .collect();
        let slots: Vec<usize> = active.iter().map(|(i, _)| *i).collect();

        StepCommand::Decode {
            token_ids,
            positions,
            slots,
            batch_size,
        }
    }

    fn process_decode_results(&mut self, reports: Vec<KimiOneTokenForwardReport>) {
        let active_slots: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|_| i))
            .collect();

        // Padding step — no active slots, no results to process
        if active_slots.is_empty() {
            return;
        }

        let mut retire = Vec::new();
        for (idx, report) in reports.into_iter().enumerate() {
            if idx >= active_slots.len() {
                break;
            }
            let slot_idx = active_slots[idx];
            let req = self.slots[slot_idx].as_mut().unwrap();
            let token_id = report.local_next_token_global_id;
            req.completion_tokens += 1;

            if req
                .token_tx
                .send(TokenEvent::Token {
                    id: token_id,
                    logprob: None,
                })
                .is_err()
            {
                retire.push(slot_idx);
                continue;
            }

            if req.completion_tokens >= req.max_tokens {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Length,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.completion_tokens,
                });
                retire.push(slot_idx);
            } else {
                req.last_token = token_id;
            }
        }

        for slot_idx in retire {
            self.slots[slot_idx] = None;
        }
    }

    fn fail_all_active(&mut self, err: &anyhow::Error) {
        let message = format!("{err:#}");
        for slot in &mut self.slots {
            if let Some(req) = slot.take() {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.completion_tokens,
                });
            }
        }
    }
}

/// Padding command for idle ranks: 1 dummy token so the rank participates
/// in EP collectives without producing real output.
fn build_padding_decode_command() -> StepCommand {
    StepCommand::Decode {
        token_ids: vec![0],
        positions: vec![0],
        slots: vec![0],
        batch_size: 1,
    }
}

fn rank_forward_loop(
    executor: Box<dyn ForwardExecutor + Send>,
    cmd_rx: std_mpsc::Receiver<StepCommand>,
    res_tx: std_mpsc::SyncSender<StepResult>,
) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            StepCommand::Decode {
                token_ids,
                positions,
                slots,
                batch_size,
            } => {
                let result =
                    executor.forward_decode_batch(&token_ids, &positions, &slots, batch_size);
                let _ = res_tx.send(StepResult::Decode(result));
            }
            StepCommand::Prefill {
                input_ids,
                slot,
                decode_batch_size,
                ep_max_seq_len,
            } => {
                let result =
                    executor.forward_prefill(&input_ids, slot, decode_batch_size, ep_max_seq_len);
                let _ = res_tx.send(StepResult::Prefill(result));
            }
            StepCommand::Shutdown => break,
        }
    }
}

fn unix_now_s() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}
