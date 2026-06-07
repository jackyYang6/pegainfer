use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use pegainfer_sim::{SimulatedEngineConfig, start_engine};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
const MODEL_NAME: &str = "pegainfer-sim-e2e";
const SERVER_START_ATTEMPTS: usize = 5;
static TEMP_MODEL_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

struct SimServer {
    base_url: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
    _model_dir: TempModelDir,
}

impl SimServer {
    async fn spawn() -> Result<Self> {
        Self::spawn_with_config(SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?).await
    }

    async fn spawn_with_config(config: SimulatedEngineConfig) -> Result<Self> {
        Self::spawn_with_model_dir_and_config(TempModelDir::with_minimal_metadata()?, config).await
    }

    async fn spawn_with_lora_routes() -> Result<Self> {
        Self::spawn_with_model_dir_and_lora_routes(TempModelDir::with_minimal_metadata()?, true)
            .await
    }

    async fn spawn_with_model_dir(model_dir: TempModelDir) -> Result<Self> {
        Self::spawn_with_model_dir_and_config(
            model_dir,
            SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?,
        )
        .await
    }

    async fn spawn_with_model_dir_and_config(
        model_dir: TempModelDir,
        config: SimulatedEngineConfig,
    ) -> Result<Self> {
        Self::spawn_with_model_dir_lora_routes_and_config(model_dir, false, config).await
    }

    async fn spawn_with_model_dir_and_lora_routes(
        model_dir: TempModelDir,
        enable_lora_routes: bool,
    ) -> Result<Self> {
        Self::spawn_with_model_dir_lora_routes_and_config(
            model_dir,
            enable_lora_routes,
            SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?,
        )
        .await
    }

    async fn spawn_with_model_dir_lora_routes_and_config(
        model_dir: TempModelDir,
        enable_lora_routes: bool,
        config: SimulatedEngineConfig,
    ) -> Result<Self> {
        let mut last_error = None;
        for attempt in 1..=SERVER_START_ATTEMPTS {
            match Self::spawn_once(&model_dir, enable_lora_routes, config).await {
                Ok(started) => {
                    return Ok(Self {
                        base_url: started.base_url,
                        shutdown: started.shutdown,
                        task: started.task,
                        _model_dir: model_dir,
                    });
                }
                Err(error) => {
                    last_error = Some(error);
                    if attempt < SERVER_START_ATTEMPTS {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("sim frontend startup was not attempted")))
            .with_context(|| {
                format!("failed to start sim frontend after {SERVER_START_ATTEMPTS} attempts")
            })
    }

    async fn spawn_once(
        model_dir: &TempModelDir,
        enable_lora_routes: bool,
        config: SimulatedEngineConfig,
    ) -> Result<StartedSimServer> {
        let port = reserve_loopback_port()?;
        let base_url = format!("http://127.0.0.1:{port}");
        let shutdown = CancellationToken::new();
        let engine = start_engine(config);
        let server_shutdown = shutdown.clone();
        let model_path = model_dir.path.to_string_lossy().into_owned();
        let mut task = tokio::spawn(async move {
            if enable_lora_routes {
                pegainfer_vllm_frontend::serve_model_with_lora_routes(
                    engine,
                    model_path,
                    vec![MODEL_NAME.to_string()],
                    Vec::new(),
                    port,
                    128,
                    server_shutdown,
                )
                .await
            } else {
                pegainfer_vllm_frontend::serve_model(
                    engine,
                    model_path,
                    vec![MODEL_NAME.to_string()],
                    port,
                    128,
                    server_shutdown,
                )
                .await
            }
        });

        let client = test_client()?;
        let health_result = tokio::select! {
            result = wait_for_health(&client, &base_url) => result,
            result = &mut task => {
                return match result {
                    Ok(Ok(())) => Err(anyhow!("sim frontend exited before becoming healthy")),
                    Ok(Err(error)) => Err(error).context("sim frontend exited before becoming healthy"),
                    Err(error) => Err(error).context("sim frontend task panicked"),
                };
            }
        };

        if let Err(error) = health_result {
            shutdown.cancel();
            let _ = tokio::time::timeout(Duration::from_secs(10), task).await;
            return Err(error);
        }

        Ok(StartedSimServer {
            base_url,
            shutdown,
            task,
        })
    }

    async fn shutdown(self) -> Result<()> {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), self.task)
            .await
            .context("timed out waiting for sim frontend shutdown")?
            .context("sim frontend task panicked")?
    }
}

struct StartedSimServer {
    base_url: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
}

struct TempModelDir {
    path: PathBuf,
}

impl TempModelDir {
    fn empty() -> Result<Self> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before Unix epoch")?
            .as_nanos();
        let sequence = TEMP_MODEL_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "pegainfer-sim-e2e-{}-{now}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path)
            .with_context(|| format!("failed to create temp model dir {}", path.display()))?;

        Ok(Self { path })
    }

    fn with_minimal_metadata() -> Result<Self> {
        let dir = Self::empty()?;

        // The simulated frontend still builds the normal vLLM text/chat stack.
        // Token-id prompts avoid tokenizer encode work, but generated ids still
        // need a tokenizer for detokenization and a tiny config for metadata.
        fs::write(dir.path.join("tokenizer.json"), TINY_TOKENIZER_JSON)
            .context("failed to write tiny tokenizer.json")?;
        fs::write(
            dir.path.join("tokenizer_config.json"),
            TINY_TOKENIZER_CONFIG_JSON,
        )
        .context("failed to write tiny tokenizer_config.json")?;
        fs::write(dir.path.join("config.json"), TINY_CONFIG_JSON)
            .context("failed to write tiny config.json")?;

        Ok(dir)
    }
}

impl Drop for TempModelDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_engine_serves_openai_completions_over_http() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_models_endpoint(&client, &server.base_url).await?;
    assert_non_streaming_completion_has_output(&client, &server.base_url).await?;
    assert_streaming_completion_emits_done(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_lora_routes_are_mounted_on_openai_frontend() -> Result<()> {
    let server = SimServer::spawn_with_lora_routes().await?;
    let client = test_client()?;

    let response = client
        .post(format!("{}/v1/load_lora_adapter", server.base_url))
        .json(&json!({
            "lora_name": "adapter-a",
            "lora_path": "/tmp/adapter-a"
        }))
        .send()
        .await?;

    let status = response.status();
    let body: Value = response.json().await?;
    if status != reqwest::StatusCode::NOT_FOUND {
        bail!("expected mounted LoRA route to report unsupported engine, got {status}: {body}");
    }
    let error = body["error"]
        .as_str()
        .ok_or_else(|| anyhow!("LoRA route response has no error string: {body}"))?;
    if !error.contains("dynamic LoRA adapter loading") {
        bail!("LoRA route returned unexpected error: {body}");
    }

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_streaming_completion_returns_nonempty_output_for_positive_max_tokens() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_non_streaming_completion_has_output(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_completion_emits_terminal_done() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_streaming_completion_emits_done(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_streaming_completion_reports_default_zero_cached_tokens() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    let completion = post_completion(&client, &server.base_url, false).await?;
    assert_usage_cached_tokens(&completion["usage"], 0)?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_streaming_completion_reports_configured_cached_tokens() -> Result<()> {
    let cached_tokens = 1;
    let server = SimServer::spawn_with_config(
        SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?
            .with_scheduled_cached_tokens(cached_tokens),
    )
    .await?;
    let client = test_client()?;

    let completion = post_completion(&client, &server.base_url, false).await?;
    assert_usage_cached_tokens(&completion["usage"], cached_tokens)?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_completion_reports_default_zero_cached_tokens_in_usage_chunk() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    let stream = post_completion_stream(&client, &server.base_url, true).await?;
    let usage = final_usage_chunk(&stream)?;
    assert_usage_cached_tokens(&usage, 0)?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_completion_reports_configured_cached_tokens_in_usage_chunk() -> Result<()> {
    let cached_tokens = 1;
    let server = SimServer::spawn_with_config(
        SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?
            .with_scheduled_cached_tokens(cached_tokens),
    )
    .await?;
    let client = test_client()?;

    let stream = post_completion_stream(&client, &server.base_url, true).await?;
    let usage = final_usage_chunk(&stream)?;
    assert_usage_cached_tokens(&usage, cached_tokens)?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_frontend_metadata_contract_is_executable() -> Result<()> {
    let model_dir = TempModelDir::with_minimal_metadata()?;
    for file in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
        if !model_dir.path.join(file).is_file() {
            bail!("minimal simulated frontend metadata fixture is missing {file}");
        }
    }

    let server = SimServer::spawn_with_model_dir(model_dir).await?;
    server.shutdown().await?;

    let error = match SimServer::spawn_with_model_dir(TempModelDir::empty()?).await {
        Ok(server) => {
            server.shutdown().await?;
            bail!("empty local model metadata directory should fail frontend startup");
        }
        Err(error) => error,
    };
    let message = format!("{error:#}");
    if !message.contains("supported tokenizer file") || !message.contains("tokenizer.json") {
        bail!("empty metadata dir failed with unexpected error: {message}");
    }

    Ok(())
}

async fn assert_models_endpoint(client: &Client, base_url: &str) -> Result<()> {
    let models: Value = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let advertised = models["data"]
        .as_array()
        .ok_or_else(|| anyhow!("/v1/models response has no data array"))?;
    if !advertised.iter().any(|model| model["id"] == MODEL_NAME) {
        bail!("/v1/models did not advertise {MODEL_NAME}: {models}");
    }

    Ok(())
}

async fn assert_non_streaming_completion_has_output(client: &Client, base_url: &str) -> Result<()> {
    let completion: Value = post_completion(client, base_url, false).await?;
    let text = completion["choices"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("non-streaming completion has no text: {completion}"))?;
    if text.is_empty() {
        bail!("non-streaming completion returned empty text for max_tokens > 0");
    }

    Ok(())
}

async fn assert_streaming_completion_emits_done(client: &Client, base_url: &str) -> Result<()> {
    let stream = post_completion_stream(client, base_url, false).await?;
    if !stream.lines().any(|line| line.trim() == "data: [DONE]") {
        bail!("streaming completion did not emit terminal data: [DONE]: {stream}");
    }

    Ok(())
}

async fn post_completion(client: &Client, base_url: &str, stream: bool) -> Result<Value> {
    let response = client
        .post(format!("{base_url}/v1/completions"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(completion_body(stream, false).to_string())
        .send()
        .await?
        .error_for_status()?;
    response
        .json()
        .await
        .context("failed to parse non-streaming completion response")
}

fn assert_usage_cached_tokens(usage: &Value, expected_cached_tokens: usize) -> Result<()> {
    let prompt_tokens = usage["prompt_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("usage has no prompt_tokens: {usage}"))?;
    if prompt_tokens != 2 {
        bail!("expected prompt_tokens=2, got {prompt_tokens}: {usage}");
    }
    let cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("usage has no prompt_tokens_details.cached_tokens: {usage}"))?;
    if cached_tokens != expected_cached_tokens as u64 {
        bail!("expected cached_tokens={expected_cached_tokens}, got {cached_tokens}: {usage}");
    }

    Ok(())
}

fn final_usage_chunk(stream: &str) -> Result<Value> {
    let mut usage_chunks = Vec::new();
    for line in stream.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(data)
            .with_context(|| format!("failed to parse SSE chunk JSON: {data}"))?;
        if !chunk["usage"].is_null() {
            usage_chunks.push(chunk["usage"].clone());
        }
    }

    match usage_chunks.as_slice() {
        [usage] => Ok(usage.clone()),
        [] => bail!("stream did not include a usage chunk: {stream}"),
        _ => bail!("stream included multiple usage chunks: {stream}"),
    }
}

async fn post_completion_stream(
    client: &Client,
    base_url: &str,
    include_usage: bool,
) -> Result<String> {
    client
        .post(format!("{base_url}/v1/completions"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(completion_body(true, include_usage).to_string())
        .send()
        .await?
        .error_for_status()?
        .text()
        .await
        .context("failed to read streaming completion response")
}

fn test_client() -> Result<Client> {
    Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .context("failed to build HTTP test client")
}

fn completion_body(stream: bool, include_usage: bool) -> Value {
    let mut body = json!({
        "model": MODEL_NAME,
        "prompt": [1, 2],
        "max_tokens": 3,
        "temperature": 0.0,
        "ignore_eos": true,
        "stream": stream
    });
    if stream && include_usage {
        body["stream_options"] = json!({ "include_usage": true });
    }
    body
}

async fn wait_for_health(client: &Client, base_url: &str) -> Result<()> {
    let health_url = format!("{base_url}/health");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if tokio::time::Instant::now() >= deadline {
            bail!("timed out waiting for sim frontend health at {health_url}");
        }

        match client
            .get(&health_url)
            .timeout(Duration::from_secs(1))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(_) | Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

fn reserve_loopback_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .context("failed to reserve loopback port for sim e2e test")?;
    Ok(listener.local_addr()?.port())
}

const TINY_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 0,
      "content": "<unk>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "alpha": 1,
      "beta": 2
    },
    "unk_token": "<unk>"
  }
}"#;

const TINY_TOKENIZER_CONFIG_JSON: &str = r#"{
  "unk_token": "<unk>",
  "tokenizer_class": "PreTrainedTokenizerFast"
}"#;

const TINY_CONFIG_JSON: &str = r#"{
  "model_type": "pegainfer_sim",
  "max_position_embeddings": 128
}"#;
