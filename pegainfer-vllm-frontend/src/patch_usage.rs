use std::collections::HashMap;
use std::sync::Arc;

use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header::CONTENT_LENGTH};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use serde::Serialize;
use tokio::sync::RwLock;
use tower::ServiceExt;

const COMPLETION_USAGE_PATCH_BODY_LIMIT: usize = 128 * 1024 * 1024;

pub(crate) type CachedTokenUsageMap = Arc<RwLock<HashMap<String, u32>>>;

#[derive(Clone)]
struct UsagePatchState {
    vllm_router: Router,
    cached_tokens_by_request_id: CachedTokenUsageMap,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompletionUsagePatchOptions {
    stream: bool,
    include_usage: bool,
}

// Rust vLLM currently emits OpenAI Usage without copying PrefillStats cached counts;
// this wrapper only patches the existing usage object with the engine-provided value.
pub(crate) fn cached_token_usage_routes(
    vllm_router: Router,
    cached_tokens_by_request_id: CachedTokenUsageMap,
) -> Router {
    Router::new()
        .route("/v1/completions", post(forward_cached_token_usage_request))
        .route(
            "/v1/chat/completions",
            post(forward_cached_token_usage_request),
        )
        .with_state(UsagePatchState {
            vllm_router: vllm_router.clone(),
            cached_tokens_by_request_id,
        })
        .fallback_service(vllm_router)
}

pub(crate) fn external_request_id(engine_request_id: &str) -> String {
    if let Some((request_id, suffix)) = engine_request_id.rsplit_once('-') {
        if suffix.len() == 8 && suffix.chars().all(|ch| ch.is_ascii_hexdigit()) {
            return request_id.to_string();
        }
    }
    engine_request_id.to_string()
}

async fn forward_cached_token_usage_request(
    State(state): State<UsagePatchState>,
    request: Request,
) -> Response {
    let (parts, body) = request.into_parts();
    let bytes = match to_bytes(body, COMPLETION_USAGE_PATCH_BODY_LIMIT).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: format!("failed to read completion request body: {error}"),
                }),
            )
                .into_response();
        }
    };
    let options = completion_request_usage_patch_options(&bytes);
    let vllm_router = state.vllm_router.clone();
    let request = Request::from_parts(parts, Body::from(bytes));
    match vllm_router.oneshot(request).await {
        Ok(response) if options.stream => {
            patch_streaming_completion_usage(response, state, options.include_usage)
        }
        Ok(response) => patch_completion_usage(response, state).await,
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("vLLM router failed to handle completion request: {error:#}"),
            }),
        )
            .into_response(),
    }
}

fn completion_request_usage_patch_options(bytes: &Bytes) -> CompletionUsagePatchOptions {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return CompletionUsagePatchOptions::default();
    };
    CompletionUsagePatchOptions {
        stream: value
            .get("stream")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        include_usage: value
            .get("stream_options")
            .and_then(|options| options.get("include_usage"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    }
}

async fn patch_completion_usage(response: Response, state: UsagePatchState) -> Response {
    let status = response.status();
    if !status.is_success() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, COMPLETION_USAGE_PATCH_BODY_LIMIT).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("failed to read completion response body: {error}"),
                }),
            )
                .into_response();
        }
    };

    let mut value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => return Response::from_parts(parts, Body::from(bytes)),
    };
    let Some(request_id) = value.get("id").and_then(serde_json::Value::as_str) else {
        return Response::from_parts(parts, Body::from(bytes));
    };
    let cached_tokens = state
        .cached_tokens_by_request_id
        .write()
        .await
        .remove(request_id)
        .unwrap_or(0);
    patch_usage_value(&mut value["usage"], cached_tokens);

    response_from_json_parts(parts, &value)
}

fn patch_streaming_completion_usage(
    response: Response,
    state: UsagePatchState,
    include_usage: bool,
) -> Response {
    let status = response.status();
    if !status.is_success() {
        return response;
    }
    let cached_tokens_by_request_id = state.cached_tokens_by_request_id;
    let (mut parts, body) = response.into_parts();
    parts.headers.remove(CONTENT_LENGTH);
    let stream = body.into_data_stream();
    let body = Body::from_stream(async_stream::stream! {
        let mut stream = stream;
        let mut request_id_for_cleanup = None;
        while let Some(next) = stream.next().await {
            let Ok(bytes) = next else {
                continue;
            };
            yield Ok::<Bytes, std::convert::Infallible>(patch_sse_usage_chunk(
                bytes,
                &cached_tokens_by_request_id,
                include_usage,
                &mut request_id_for_cleanup,
            ).await);
        }
        if let Some(request_id) = request_id_for_cleanup {
            cached_tokens_by_request_id.write().await.remove(&request_id);
        }
    });
    Response::from_parts(parts, body)
}

async fn patch_sse_usage_chunk(
    bytes: Bytes,
    cached_tokens_by_request_id: &CachedTokenUsageMap,
    include_usage: bool,
    request_id_for_cleanup: &mut Option<String>,
) -> Bytes {
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return bytes;
    };
    let mut patched = String::with_capacity(text.len());
    let mut changed = false;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            patched.push_str(line);
            patched.push('\n');
            continue;
        };
        if data.trim() == "[DONE]" {
            patched.push_str(line);
            patched.push('\n');
            continue;
        }
        let Ok(mut value) = serde_json::from_str::<serde_json::Value>(data) else {
            patched.push_str(line);
            patched.push('\n');
            continue;
        };
        if request_id_for_cleanup.is_none() {
            if let Some(request_id) = value.get("id").and_then(serde_json::Value::as_str) {
                *request_id_for_cleanup = Some(request_id.to_string());
            }
        }
        if include_usage && value.get("usage").is_some_and(|usage| !usage.is_null()) {
            if let Some(request_id) = value.get("id").and_then(serde_json::Value::as_str) {
                let cached_tokens = cached_tokens_by_request_id
                    .write()
                    .await
                    .remove(request_id)
                    .unwrap_or(0);
                patch_usage_value(&mut value["usage"], cached_tokens);
                patched.push_str("data: ");
                patched.push_str(&value.to_string());
                patched.push('\n');
                changed = true;
                continue;
            }
        }
        patched.push_str(line);
        patched.push('\n');
    }
    if changed { Bytes::from(patched) } else { bytes }
}

fn patch_usage_value(usage: &mut serde_json::Value, cached_tokens: u32) {
    let Some(usage) = usage.as_object_mut() else {
        return;
    };
    let details = usage
        .entry("prompt_tokens_details")
        .or_insert_with(|| serde_json::json!({}));
    if !details.is_object() {
        *details = serde_json::json!({});
    }
    details
        .as_object_mut()
        .expect("prompt_tokens_details must be object")
        .insert(
            "cached_tokens".to_string(),
            serde_json::Value::Number(cached_tokens.into()),
        );
}

fn response_from_json_parts(
    mut parts: axum::http::response::Parts,
    value: &serde_json::Value,
) -> Response {
    let body = value.to_string();
    parts.headers.insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&body.len().to_string()).expect("json body length must be valid"),
    );
    Response::from_parts(parts, Body::from(body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn chat_completion_usage_is_patched_with_cached_tokens() {
        let cached_tokens_by_request_id =
            Arc::new(RwLock::new(HashMap::from([("chatcmpl-1".to_string(), 7)])));
        let vllm_router = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                Json(serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "usage": {
                        "prompt_tokens": 11,
                        "prompt_tokens_details": {}
                    }
                }))
            }),
        );
        let router = cached_token_usage_routes(vllm_router, cached_tokens_by_request_id.clone());
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .body(Body::from(
                serde_json::json!({
                    "model": "model",
                    "messages": [{"role": "user", "content": "hello"}]
                })
                .to_string(),
            ))
            .expect("request");

        let response = router.oneshot(request).await.expect("route request");

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), COMPLETION_USAGE_PATCH_BODY_LIMIT)
            .await
            .expect("read body");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("json response");
        assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
        assert!(cached_tokens_by_request_id.read().await.is_empty());
    }

    #[tokio::test]
    async fn streaming_chat_completion_usage_chunk_is_patched_when_included() {
        let cached_tokens_by_request_id = Arc::new(RwLock::new(HashMap::from([(
            "chatcmpl-stream".to_string(),
            5,
        )])));
        let vllm_router = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                Response::builder()
                    .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from(
                        "data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"usage\":null}\n\
                         data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"usage\":{\"prompt_tokens\":11,\"prompt_tokens_details\":{}}}\n\
                         data: [DONE]\n",
                    ))
                    .expect("streaming response")
            }),
        );
        let router = cached_token_usage_routes(vllm_router, cached_tokens_by_request_id.clone());
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .body(Body::from(
                serde_json::json!({
                    "model": "model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": true,
                    "stream_options": {"include_usage": true}
                })
                .to_string(),
            ))
            .expect("request");

        let response = router.oneshot(request).await.expect("route request");

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), COMPLETION_USAGE_PATCH_BODY_LIMIT)
            .await
            .expect("read body");
        let text = std::str::from_utf8(&bytes).expect("utf8 stream");
        let usage = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| data.trim() != "[DONE]")
            .map(|data| serde_json::from_str::<serde_json::Value>(data).expect("sse json"))
            .find_map(|value| value["usage"].is_object().then(|| value["usage"].clone()))
            .expect("usage chunk");
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 5);
        assert!(cached_tokens_by_request_id.read().await.is_empty());
    }
}
