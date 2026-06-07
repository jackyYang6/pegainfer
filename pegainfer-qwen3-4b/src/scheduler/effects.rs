use log::debug;
use tokio::sync::mpsc;

use crate::executor::RequestId;
use pegainfer_core::engine::{FinishReason, TokenLogprob};

use super::{ActiveRequestState, TokenEvent};

pub(super) struct PromptEchoEffect {
    pub(super) ids: Vec<u32>,
    pub(super) logprobs: Vec<Option<TokenLogprob>>,
}

#[derive(Clone, Copy)]
pub(super) struct ScheduledEffect {
    pub(super) queued_at_unix_s: f64,
    pub(super) scheduled_at_unix_s: f64,
    pub(super) prompt_tokens: usize,
    pub(super) cached_tokens: usize,
}

pub(super) enum PendingEffect {
    Finish {
        request_id: RequestId,
        token_tx: mpsc::UnboundedSender<TokenEvent>,
        scheduled: ScheduledEffect,
        prompt_echo: Option<PromptEchoEffect>,
        finish_reason: FinishReason,
        completion_tokens: usize,
    },
    EmitAndFinish {
        request_id: RequestId,
        token_tx: mpsc::UnboundedSender<TokenEvent>,
        scheduled: ScheduledEffect,
        prompt_echo: Option<PromptEchoEffect>,
        token: u32,
        logprob: Option<TokenLogprob>,
        finish_reason: FinishReason,
        completion_tokens: usize,
    },
    Promote {
        state: ActiveRequestState,
        scheduled: ScheduledEffect,
        prompt_echo: Option<PromptEchoEffect>,
        first_token: u32,
        logprob: Option<TokenLogprob>,
    },
}

pub(super) enum DecodeEffect {
    Finish {
        request_id: RequestId,
        finish_reason: FinishReason,
        completion_tokens: usize,
    },
    EmitAndFinish {
        request_id: RequestId,
        token: u32,
        logprob: Option<TokenLogprob>,
        finish_reason: FinishReason,
        completion_tokens: usize,
    },
    EmitAndContinue {
        request_id: RequestId,
        token: u32,
        logprob: Option<TokenLogprob>,
        completion_tokens: usize,
    },
}

pub(super) struct StepEffects {
    pub(super) pending: Vec<PendingEffect>,
    pub(super) decode: Vec<DecodeEffect>,
}

#[derive(Default)]
pub(super) struct PrefixCacheStats {
    total_requests: u64,
    hit_requests: u64,
    miss_requests: u64,
    total_prompt_tokens: u64,
    total_cached_tokens: u64,
}

impl PrefixCacheStats {
    pub(super) fn observe(&mut self, scheduled: ScheduledEffect) {
        self.total_requests += 1;
        if scheduled.cached_tokens > 0 {
            self.hit_requests += 1;
        } else {
            self.miss_requests += 1;
        }
        self.total_prompt_tokens += scheduled.prompt_tokens as u64;
        self.total_cached_tokens += scheduled.cached_tokens as u64;
    }

    pub(super) fn log_snapshot(&self) {
        debug!(
            "Qwen3 prefix cache stats: total_requests={}, hit_requests={}, miss_requests={}, hit_rate={:.4}, total_prompt_tokens={}, total_cached_tokens={}, token_hit_rate={:.4}",
            self.total_requests,
            self.hit_requests,
            self.miss_requests,
            self.hit_rate(),
            self.total_prompt_tokens,
            self.total_cached_tokens,
            self.token_hit_rate()
        );
    }

    fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.hit_requests as f64 / self.total_requests as f64
    }

    fn token_hit_rate(&self) -> f64 {
        if self.total_prompt_tokens == 0 {
            return 0.0;
        }
        self.total_cached_tokens as f64 / self.total_prompt_tokens as f64
    }

    #[cfg(test)]
    pub(super) fn totals(&self) -> (u64, u64, u64, f64, u64, u64, f64) {
        (
            self.total_requests,
            self.hit_requests,
            self.miss_requests,
            self.hit_rate(),
            self.total_prompt_tokens,
            self.total_cached_tokens,
            self.token_hit_rate(),
        )
    }
}

impl StepEffects {
    pub(super) fn empty() -> Self {
        Self {
            pending: Vec::new(),
            decode: Vec::new(),
        }
    }
}

fn send_pending_scheduled_and_echo(
    token_tx: &mpsc::UnboundedSender<TokenEvent>,
    scheduled: ScheduledEffect,
    prompt_echo: Option<PromptEchoEffect>,
    prefix_cache_stats: &mut PrefixCacheStats,
) -> Result<(), mpsc::error::SendError<TokenEvent>> {
    prefix_cache_stats.observe(scheduled);
    prefix_cache_stats.log_snapshot();
    token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s: scheduled.queued_at_unix_s,
        scheduled_at_unix_s: scheduled.scheduled_at_unix_s,
        prompt_tokens: scheduled.prompt_tokens,
        cached_tokens: scheduled.cached_tokens,
    })?;
    if let Some(echo) = prompt_echo {
        token_tx.send(TokenEvent::PromptTokens {
            ids: echo.ids,
            logprobs: echo.logprobs,
        })?;
    }
    Ok(())
}

pub(super) fn apply_effects(
    executor: &mut impl crate::executor::ModelExecutor,
    active: &mut Vec<ActiveRequestState>,
    effects: StepEffects,
    prefix_cache_stats: &mut PrefixCacheStats,
) {
    let mut to_retire = Vec::new();
    for effect in effects.decode {
        match effect {
            DecodeEffect::Finish {
                request_id,
                finish_reason,
                completion_tokens,
            } => {
                let Some(index) = active.iter().position(|req| req.request_id == request_id) else {
                    continue;
                };
                let req = &active[index];
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens: req.prompt_len,
                    completion_tokens,
                });
                let _ = executor.drop_request(request_id);
                to_retire.push(index);
            }
            DecodeEffect::EmitAndFinish {
                request_id,
                token,
                logprob,
                finish_reason,
                completion_tokens,
            } => {
                let Some(index) = active.iter().position(|req| req.request_id == request_id) else {
                    continue;
                };
                let req = &active[index];
                if req
                    .token_tx
                    .send(TokenEvent::Token { id: token, logprob })
                    .is_ok()
                {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason,
                        prompt_tokens: req.prompt_len,
                        completion_tokens,
                    });
                }
                let _ = executor.drop_request(request_id);
                to_retire.push(index);
            }
            DecodeEffect::EmitAndContinue {
                request_id,
                token,
                logprob,
                completion_tokens,
            } => {
                let Some(index) = active.iter().position(|req| req.request_id == request_id) else {
                    continue;
                };
                let req = &mut active[index];
                if req
                    .token_tx
                    .send(TokenEvent::Token { id: token, logprob })
                    .is_err()
                {
                    let _ = executor.drop_request(request_id);
                    to_retire.push(index);
                } else {
                    req.last_token = token;
                    req.generated_count = completion_tokens;
                }
            }
        }
    }
    to_retire.sort_unstable();
    to_retire.dedup();
    for &i in to_retire.iter().rev() {
        active.swap_remove(i);
    }

    for effect in effects.pending {
        match effect {
            PendingEffect::Finish {
                request_id,
                token_tx,
                scheduled,
                prompt_echo,
                finish_reason,
                completion_tokens,
            } => {
                let _ = send_pending_scheduled_and_echo(
                    &token_tx,
                    scheduled,
                    prompt_echo,
                    prefix_cache_stats,
                );
                let _ = token_tx.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens: scheduled.prompt_tokens,
                    completion_tokens,
                });
                let _ = executor.drop_request(request_id);
            }
            PendingEffect::EmitAndFinish {
                request_id,
                token_tx,
                scheduled,
                prompt_echo,
                token,
                logprob,
                finish_reason,
                completion_tokens,
            } => {
                if send_pending_scheduled_and_echo(
                    &token_tx,
                    scheduled,
                    prompt_echo,
                    prefix_cache_stats,
                )
                .is_ok()
                    && token_tx
                        .send(TokenEvent::Token { id: token, logprob })
                        .is_ok()
                {
                    let _ = token_tx.send(TokenEvent::Finished {
                        finish_reason,
                        prompt_tokens: scheduled.prompt_tokens,
                        completion_tokens,
                    });
                }
                let _ = executor.drop_request(request_id);
            }
            PendingEffect::Promote {
                state,
                scheduled,
                prompt_echo,
                first_token,
                logprob,
            } => {
                if send_pending_scheduled_and_echo(
                    &state.token_tx,
                    scheduled,
                    prompt_echo,
                    prefix_cache_stats,
                )
                .is_ok()
                    && state
                        .token_tx
                        .send(TokenEvent::Token {
                            id: first_token,
                            logprob,
                        })
                        .is_ok()
                {
                    active.push(state);
                } else {
                    let _ = executor.drop_request(state.request_id);
                }
            }
        }
    }
}
