use std::{
    env,
    hash::{DefaultHasher, Hash, Hasher},
    sync::LazyLock,
    vec,
};

use axum::{
    Json,
    extract::{FromRequest, Request},
};
use http::HeaderMap;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    config::{CLAUDE_CODE_BILLING_SALT, CLAUDE_CODE_VERSION, CLEWDR_CONFIG},
    error::ClewdrError,
    middleware::claude::{ClaudeApiFormat, ClaudeContext},
    types::{
        claude::{
            ContentBlock, CreateMessageParams, Message, MessageContent, OutputConfig,
            OutputEffort, Role, Thinking, Usage,
        },
        oai::CreateMessageParams as OaiCreateMessageParams,
    },
};

/// A custom extractor that unifies different API formats
///
/// This extractor processes incoming requests, handling differences between
/// Claude and OpenAI API formats, and applies preprocessing to ensure consistent
/// handling throughout the application. It also detects and handles test messages
/// from client applications.
///
/// # Functionality
///
/// - Extracts and normalizes message parameters from different API formats
/// - Detects and processes "thinking mode" requests by modifying model names
/// - Identifies test messages and handles them appropriately
/// - Attempts to retrieve responses from cache before processing requests
/// - Provides format information via the FormatInfo extension
pub struct ClaudeWebPreprocess(pub CreateMessageParams, pub ClaudeContext);

/// Contains information about the API format and streaming status
///
/// This structure is passed through the request pipeline to inform
/// handlers and response processors about the API format being used
/// and whether the response should be streamed.
#[derive(Debug, Clone)]
pub struct ClaudeWebContext {
    /// Whether the response should be streamed
    pub(super) stream: bool,
    /// The API format being used (Claude or OpenAI)
    pub(super) api_format: ClaudeApiFormat,
    /// The stop sequence used for the request
    pub(super) stop_sequences: Vec<String>,
    /// User information about input and output tokens
    pub(super) usage: Usage,
}

/// Predefined test message in Claude format for connection testing
///
/// This is a standard test message sent by clients like SillyTavern
/// to verify connectivity. The system detects these messages and
/// responds with a predefined test response to confirm service availability.
static TEST_MESSAGE_CLAUDE: LazyLock<Message> =
    LazyLock::new(|| Message::new_blocks(Role::User, vec![ContentBlock::text("Hi")]));

/// Predefined test message in OpenAI format for connection testing
static TEST_MESSAGE_OAI: LazyLock<Message> = LazyLock::new(|| Message::new_text(Role::User, "Hi"));

struct NormalizeRequest(CreateMessageParams, ClaudeApiFormat);

const CLAUDE_CODE_ENTRYPOINT_ENV: &str = "CLAUDE_CODE_ENTRYPOINT";

fn prepend_system_blocks(body: &mut CreateMessageParams, blocks: Vec<ContentBlock>) {
    if blocks.is_empty() {
        return;
    }

    let mut prefixed = blocks
        .into_iter()
        .map(|block| json!(block))
        .collect::<Vec<_>>();
    match body.system.take() {
        Some(Value::String(text)) if !text.trim().is_empty() => {
            prefixed.push(json!(ContentBlock::text(text)));
        }
        Some(Value::Array(mut systems)) => {
            prefixed.append(&mut systems);
        }
        Some(Value::Null) | None => {}
        Some(other) => prefixed.push(other),
    }
    body.system = Some(Value::Array(prefixed));
}

fn first_user_message_text(messages: &[Message]) -> &str {
    messages
        .iter()
        .find(|message| message.role == Role::User)
        .and_then(|message| match &message.content {
            MessageContent::Text { content } => Some(content.as_str()),
            MessageContent::Blocks { content } => content.iter().find_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            }),
        })
        .unwrap_or_default()
}

fn sample_js_code_unit(text: &str, idx: usize) -> String {
    text.encode_utf16()
        .nth(idx)
        .map(|unit| String::from_utf16_lossy(&[unit]))
        .unwrap_or_else(|| "0".to_string())
}

fn claude_code_billing_header(messages: &[Message]) -> String {
    let sampled = [4, 7, 20]
        .into_iter()
        .map(|idx| sample_js_code_unit(first_user_message_text(messages), idx))
        .collect::<String>();
    let version_hash = hex::encode(Sha256::digest(format!(
        "{CLAUDE_CODE_BILLING_SALT}{sampled}{CLAUDE_CODE_VERSION}"
    )));
    let entrypoint = env::var(CLAUDE_CODE_ENTRYPOINT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "cli".to_string());

    format!(
        "x-anthropic-billing-header: cc_version={CLAUDE_CODE_VERSION}.{}; cc_entrypoint={entrypoint}; cch=00000;",
        &version_hash[..3]
    )
}

fn drop_empty_system(body: &mut CreateMessageParams) {
    let Some(system) = body.system.take() else {
        return;
    };

    let is_empty = match &system {
        Value::Null => true,
        Value::String(text) => text.trim().is_empty(),
        Value::Array(systems) => systems.is_empty()
            || systems.iter().all(|entry| match entry {
                Value::Null => true,
                Value::String(text) => text.trim().is_empty(),
                Value::Object(obj) if matches!(obj.get("type"), Some(Value::String(t)) if t == "text") => {
                    obj.get("text")
                        .and_then(Value::as_str)
                        .is_none_or(|text| text.trim().is_empty())
                }
                _ => false,
            }),
        _ => false,
    };

    body.system = (!is_empty).then_some(system);
}

fn strip_ephemeral_scope_from_system(system: &mut Value) {
    let Some(items) = system.as_array_mut() else {
        return;
    };

    for item in items {
        let Some(obj) = item.as_object_mut() else {
            continue;
        };
        let Some(cache_control) = obj.get_mut("cache_control") else {
            continue;
        };
        let Some(cache_obj) = cache_control.as_object_mut() else {
            continue;
        };

        if let Some(ephemeral) = cache_obj.get_mut("ephemeral")
            && let Some(ephemeral_obj) = ephemeral.as_object_mut()
        {
            ephemeral_obj.remove("scope");
        }

        if matches!(cache_obj.get("type"), Some(Value::String(t)) if t == "ephemeral") {
            cache_obj.remove("scope");
        }
    }
}

fn extract_anthropic_beta_header(headers: &HeaderMap) -> Option<String> {
    let mut parts = Vec::new();
    for value in headers.get_all("anthropic-beta") {
        if let Ok(raw) = value.to_str() {
            for token in raw.split(',') {
                let token = token.trim();
                if !token.is_empty() {
                    parts.push(token.to_string());
                }
            }
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    }
}

/// Modern strict families: reject non-default sampling (temperature/top_p/
/// top_k 400) and default to `Thinking::Adaptive` on the -thinking suffix.
/// Opus 4.7/4.8 are adaptive-ONLY (enabled+budget 400s); Fable 5 accepts
/// both thinking modes but adaptive+summarized is the right default.
fn is_adaptive_only_family(model: &str) -> bool {
    model.starts_with("claude-opus-4-7")
        || model.starts_with("claude-opus-4-8")
        || model.starts_with("claude-fable-5")
        || model.starts_with("claude-sonnet-5")
}

/// Families that DO NOT accept `output_config.effort` (per Anthropic docs).
/// For these we back-fill `Thinking::Enabled { budget_tokens }` from the OAI
/// effort hint so OAI clients keep getting extended reasoning, and clear
/// `output_config` since it would be ignored upstream anyway.
///
/// Models that DO accept effort: Opus 4.5 / 4.6 / 4.7 / 4.8, Sonnet 4.6, Mythos.
/// Models that DO NOT: 3.7 Sonnet, Sonnet 4 (20250514), Sonnet 4.5, Opus 4
/// (20250514), Opus 4.1.
fn is_pre_effort_family(model: &str) -> bool {
    model.starts_with("claude-3-7-sonnet")
        || model.starts_with("claude-sonnet-4-20")
        || model.starts_with("claude-sonnet-4-5")
        || model.starts_with("claude-opus-4-20")
        || model.starts_with("claude-opus-4-1-")
}

fn effort_to_budget(e: &OutputEffort) -> u64 {
    match e {
        OutputEffort::Low => 1024,
        OutputEffort::Medium => 4096,
        OutputEffort::High => 16384,
        OutputEffort::Max => 32768,
    }
}

/// Per-family wire-shape fixes applied once at NormalizeRequest time.
/// Covers every caller (OAI translation, native Claude passthrough,
/// -thinking suffix path).
fn apply_model_family_fixes(body: &mut CreateMessageParams) {
    let model = body.model.as_str();

    // Opus 4.7/4.8 specifics.
    if is_adaptive_only_family(model) {
        // Anthropic 400s on non-default sampling for 4.7/4.8.
        body.temperature = None;
        body.top_p = None;
        body.top_k = None;
        // On 4.7/4.8 the API default for `thinking.display` is `omitted` —
        // empty thinking blocks. Force `summarized` whenever adaptive
        // thinking is on so client thinking UIs (SillyTavern, JAI) populate.
        if let Some(Thinking::Adaptive { display }) = body.thinking.as_mut()
            && display.is_none()
        {
            *display = Some("summarized".to_string());
        }
    }

    // Effort handling per family.
    if is_pre_effort_family(model) {
        // Pre-effort families ignore output_config. Back-fill thinking from
        // any effort hint so OAI clients sending reasoning_effort still get
        // extended reasoning, then clear output_config so it's not sent.
        if body.thinking.is_none()
            && let Some(effort) = body.output_config.as_ref().and_then(|c| c.effort.as_ref())
        {
            body.thinking = Some(Thinking::new(effort_to_budget(effort)));
        }
        body.output_config = None;
    } else {
        // All families that accept output_config.effort (Opus 4.5/4.6/4.7/4.8,
        // Sonnet 4.6, Mythos): default to Max when caller sent none.
        // Caller-supplied effort always wins.
        let cfg = body.output_config.get_or_insert(OutputConfig {
            effort: None,
            format: None,
        });
        if cfg.effort.is_none() {
            cfg.effort = Some(OutputEffort::Max);
        }
    }
}

fn sanitize_messages(msgs: Vec<Message>) -> Vec<Message> {
    msgs.into_iter()
        .filter_map(|m| {
            let role = m.role;
            let content = match m.content {
                MessageContent::Text { content } => {
                    let trimmed = content.trim().to_string();
                    if role == Role::Assistant && trimmed.is_empty() {
                        return None;
                    }
                    MessageContent::Text { content: trimmed }
                }
                MessageContent::Blocks { content } => {
                    let new_blocks: Vec<ContentBlock> = content
                        .into_iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text, .. } => {
                                let t = text.trim().to_string();
                                if t.is_empty() {
                                    None
                                } else {
                                    Some(ContentBlock::text(t))
                                }
                            }
                            other => Some(other),
                        })
                        .collect();
                    if role == Role::Assistant && new_blocks.is_empty() {
                        return None;
                    }
                    MessageContent::Blocks {
                        content: new_blocks,
                    }
                }
            };
            Some(Message { role, content })
        })
        .collect()
}

impl<S> FromRequest<S> for NormalizeRequest
where
    S: Send + Sync,
{
    type Rejection = ClewdrError;

    async fn from_request(req: Request, _: &S) -> Result<Self, Self::Rejection> {
        let uri = req.uri().to_string();
        let format = if uri.contains("chat/completions") {
            ClaudeApiFormat::OpenAI
        } else {
            ClaudeApiFormat::Claude
        };
        let Json(mut body) = match format {
            ClaudeApiFormat::OpenAI => {
                let Json(json) = Json::<OaiCreateMessageParams>::from_request(req, &()).await?;
                Json(json.into())
            }
            ClaudeApiFormat::Claude => Json::<CreateMessageParams>::from_request(req, &()).await?,
        };
        if CLEWDR_CONFIG.load().sanitize_messages {
            // Trim whitespace and drop empty assistant turns when enabled.
            body.messages = sanitize_messages(body.messages);
        }
        if body.model.ends_with("-thinking") {
            body.model = body.model.trim_end_matches("-thinking").to_string();
            let default = if is_adaptive_only_family(&body.model) {
                Thinking::Adaptive {
                    display: Some("summarized".to_string()),
                }
            } else {
                // Derive budget from effort hint if present (e.g. OAI reasoning_effort),
                // else fall back to the default 4096 budget.
                let budget = body
                    .output_config
                    .as_ref()
                    .and_then(|c| c.effort.as_ref())
                    .map(effort_to_budget)
                    .unwrap_or(4096);
                Thinking::new(budget)
            };
            body.thinking.get_or_insert(default);
        }
        apply_model_family_fixes(&mut body);
        drop_empty_system(&mut body);
        Ok(Self(body, format))
    }
}

impl<S> FromRequest<S> for ClaudeWebPreprocess
where
    S: Send + Sync,
{
    type Rejection = ClewdrError;

    async fn from_request(req: Request, _: &S) -> Result<Self, Self::Rejection> {
        let NormalizeRequest(body, format) = NormalizeRequest::from_request(req, &()).await?;

        // Check for test messages and respond appropriately
        if !body.stream.unwrap_or_default()
            && (body.messages == vec![TEST_MESSAGE_CLAUDE.to_owned()]
                || body.messages == vec![TEST_MESSAGE_OAI.to_owned()])
        {
            // Respond with a test message
            return Err(ClewdrError::TestMessage);
        }

        // Determine streaming status and API format
        let stream = body.stream.unwrap_or_default();

        let input_tokens = body.count_tokens();
        let info = ClaudeWebContext {
            stream,
            api_format: format,
            stop_sequences: body.stop_sequences.to_owned().unwrap_or_default(),
            usage: Usage {
                input_tokens,
                output_tokens: 0, // Placeholder for output token count
            },
        };

        Ok(Self(body, ClaudeContext::Web(info)))
    }
}

#[derive(Debug, Clone)]
pub struct ClaudeCodeContext {
    /// Whether the response should be streamed
    pub(super) stream: bool,
    /// The API format being used (Claude or OpenAI)
    pub(super) api_format: ClaudeApiFormat,
    /// The hash of the system messages for caching purposes
    pub(super) system_prompt_hash: Option<u64>,
    /// Optional anthropic-beta header forwarded from client request
    pub(super) anthropic_beta: Option<String>,
    // Usage information for the request
    pub(super) usage: Usage,
}

pub struct ClaudeCodePreprocess(pub CreateMessageParams, pub ClaudeContext);

impl<S> FromRequest<S> for ClaudeCodePreprocess
where
    S: Send + Sync,
{
    type Rejection = ClewdrError;

    async fn from_request(req: Request, _: &S) -> Result<Self, Self::Rejection> {
        let anthropic_beta = extract_anthropic_beta_header(req.headers());
        let NormalizeRequest(mut body, format) = NormalizeRequest::from_request(req, &()).await?;
        // Handle thinking mode by modifying the model name
        if body.temperature.is_some() {
            body.top_p = None; // temperature and top_p cannot be used together in Opus-4.x
        }

        // Check for test messages and respond appropriately
        if !body.stream.unwrap_or_default()
            && (body.messages == vec![TEST_MESSAGE_CLAUDE.to_owned()]
                || body.messages == vec![TEST_MESSAGE_OAI.to_owned()])
        {
            // Respond with a test message
            return Err(ClewdrError::TestMessage);
        }

        // Determine streaming status and API format
        let stream = body.stream.unwrap_or_default();

        let mut system_prefixes = vec![ContentBlock::text(claude_code_billing_header(
            &body.messages,
        ))];
        if let Some(custom_system) = CLEWDR_CONFIG
            .load()
            .custom_system
            .clone()
            .filter(|s| !s.trim().is_empty())
        {
            system_prefixes.push(ContentBlock::text(custom_system));
        }
        prepend_system_blocks(&mut body, system_prefixes);

        if let Some(system) = body.system.as_mut() {
            strip_ephemeral_scope_from_system(system);
        }

        let cache_systems = body
            .system
            .as_ref()
            .and_then(Value::as_array)
            .map(|systems| {
                systems
                    .iter()
                    .filter(|s| s["cache_control"].as_object().is_some())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let system_prompt_hash = (!cache_systems.is_empty()).then(|| {
            let mut hasher = DefaultHasher::new();
            cache_systems.hash(&mut hasher);
            hasher.finish()
        });

        let input_tokens = body.count_tokens();

        let info = ClaudeCodeContext {
            stream,
            api_format: format,
            system_prompt_hash,
            anthropic_beta,
            usage: Usage {
                input_tokens,
                output_tokens: 0, // Placeholder for output token count
            },
        };

        Ok(Self(body, ClaudeContext::Code(info)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_code_billing_header_matches_2176_rule() {
        let messages = vec![Message::new_text(Role::User, "hey")];

        assert_eq!(
            claude_code_billing_header(&messages),
            "x-anthropic-billing-header: cc_version=2.1.76.4dc; cc_entrypoint=cli; cch=00000;"
        );
    }

    #[test]
    fn claude_code_billing_header_uses_first_text_block_of_first_user_message() {
        let messages = vec![
            Message::new_blocks(
                Role::User,
                vec![
                    ContentBlock::Image {
                        source: crate::types::claude::ImageSource::Url {
                            url: "https://example.com/a.png".to_string(),
                        },
                        cache_control: None,
                    },
                    ContentBlock::text("abcdefg"),
                    ContentBlock::text("ignored"),
                ],
            ),
            Message::new_text(Role::User, "later"),
        ];

        assert_eq!(
            claude_code_billing_header(&messages),
            "x-anthropic-billing-header: cc_version=2.1.76.540; cc_entrypoint=cli; cch=00000;"
        );
    }

    #[test]
    fn prepend_system_blocks_keeps_billing_before_custom_system() {
        let mut body = CreateMessageParams {
            messages: vec![Message::new_text(Role::User, "hey")],
            model: "claude-sonnet-4-5".to_string(),
            system: Some(json!("original system")),
            ..Default::default()
        };

        prepend_system_blocks(
            &mut body,
            vec![
                ContentBlock::text("billing"),
                ContentBlock::text("custom system"),
            ],
        );

        let systems = body.system.unwrap().as_array().cloned().unwrap();
        let texts = systems
            .iter()
            .map(|value| value["text"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(texts, vec!["billing", "custom system", "original system"]);
    }
}
