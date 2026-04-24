//! SSE streaming converter: OpenAI ChatCompletions SSE → Claude SSE events.
//!
//! Consumes an upstream stream of [`OpenAiSseEvent`] items and re-emits them
//! as Claude-compatible SSE events, maintaining a state machine for tool call
//! accumulation (I2 strategy: buffer all argument fragments, emit one merged
//! `input_json_delta` at `finish_reason=tool_calls`).

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::time::Duration;

use axum::response::sse::Event;
use futures::stream::Stream;
use futures::StreamExt;
use serde_json::json;
use tokio::time::timeout;
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::types::claude::{sse, stop_reason, Usage};
use crate::types::openai::ChatCompletionChunk;
use crate::util::{fix_json, tool_id, tool_name};

// ---------------------------------------------------------------------------
// Public types re-exported for client.rs
// ---------------------------------------------------------------------------

/// A parsed SSE event from the ChatCompletions stream.
#[derive(Debug)]
pub enum OpenAiSseEvent {
    /// A chunk of streaming data.
    Chunk(ChatCompletionChunk),
    /// The `[DONE]` sentinel.
    Done,
}

/// Error type for SSE stream parsing.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("connection error: {0}")]
    Connection(String),
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum Phase {
    /// Initial — waiting for first chunk (message_start not yet emitted).
    AwaitingFirstChunk,
    /// Streaming — message_start already emitted, processing chunks.
    Streaming,
    /// Emit closing events.
    Epilogue,
    /// Terminal.
    Done,
}

/// Per-tool-call accumulation state.
struct ToolCallAccumulator {
    id: String,
    name: String,
    args: String,
}

struct ConverterState<S> {
    upstream: Pin<Box<S>>,
    original_model: String,

    block_index: usize,
    has_tool_call: bool,
    /// Accumulated tool calls keyed by the chunk `index` field.
    tool_calls: HashMap<usize, ToolCallAccumulator>,

    final_stop_reason: String,
    usage: Usage,

    phase: Phase,
    idle_timeout: Duration,
    estimated_input_tokens: u32,
    tool_name_map: tool_name::ToolNameMap,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert a ChatCompletions streaming SSE sequence into a Claude-compatible
/// SSE event stream.
pub fn openai_stream_to_claude(
    upstream: impl Stream<Item = Result<OpenAiSseEvent, StreamError>> + Send + 'static,
    original_model: String,
    idle_timeout: Duration,
    estimated_input_tokens: u32,
    tool_name_map: tool_name::ToolNameMap,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> + Send {
    let state = ConverterState {
        upstream: Box::pin(upstream),
        original_model,
        block_index: 0,
        has_tool_call: false,
        tool_calls: HashMap::new(),
        final_stop_reason: stop_reason::END_TURN.to_string(),
        usage: Usage::default(),
        phase: Phase::AwaitingFirstChunk,
        idle_timeout,
        estimated_input_tokens,
        tool_name_map,
    };

    futures::stream::unfold(
        (state, VecDeque::<Event>::new()),
        |(mut state, mut buf)| async move {
            loop {
                if let Some(event) = buf.pop_front() {
                    return Some((Ok(event), (state, buf)));
                }

                match state.phase {
                    Phase::AwaitingFirstChunk | Phase::Streaming => {
                        let next = if state.idle_timeout.is_zero() {
                            state.upstream.next().await
                        } else {
                            match timeout(state.idle_timeout, state.upstream.next()).await {
                                Ok(result) => result,
                                Err(_) => {
                                    let kind = if state.phase == Phase::AwaitingFirstChunk {
                                        "first-byte"
                                    } else {
                                        "idle"
                                    };
                                    error!(
                                        "stream converter {kind} timeout ({}s)",
                                        state.idle_timeout.as_secs()
                                    );
                                    emit_error_event(
                                        &format!(
                                            "stream {kind} timeout ({}s)",
                                            state.idle_timeout.as_secs()
                                        ),
                                        &mut buf,
                                    );
                                    state.phase = Phase::Epilogue;
                                    continue;
                                }
                            }
                        };

                        match next {
                            Some(Ok(event)) => {
                                process_event(&mut state, event, &mut buf);
                            }
                            Some(Err(e)) => {
                                error!("upstream stream error: {e}");
                                emit_error_event(&e.to_string(), &mut buf);
                                state.phase = Phase::Epilogue;
                            }
                            None => {
                                // Stream ended without [DONE].
                                if state.phase == Phase::AwaitingFirstChunk {
                                    warn!("stream ended without any chunks");
                                }
                                state.phase = Phase::Epilogue;
                            }
                        }
                    }
                    Phase::Epilogue => {
                        emit_epilogue(&state, &mut buf);
                        state.phase = Phase::Done;
                    }
                    Phase::Done => {
                        return None;
                    }
                }
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Event processing
// ---------------------------------------------------------------------------

fn process_event(
    state: &mut ConverterState<impl Stream>,
    event: OpenAiSseEvent,
    buf: &mut VecDeque<Event>,
) {
    match event {
        OpenAiSseEvent::Done => {
            // Flush accumulated tool calls before epilogue.
            flush_tool_calls(state, buf);
            state.phase = Phase::Epilogue;
        }
        OpenAiSseEvent::Chunk(chunk) => {
            let msg_id = &chunk.id;

            // Usage from the last chunk (stream_options.include_usage).
            if let Some(u) = &chunk.usage {
                state.usage.input_tokens = u.prompt_tokens;
                state.usage.output_tokens = u.completion_tokens;
                if let Some(ref details) = u.prompt_tokens_details {
                    if let Some(cached) = details.cached_tokens {
                        if cached > 0 {
                            state.usage.cache_read_input_tokens = Some(cached);
                        }
                    }
                }
            }

            let Some(choice) = chunk.choices.first() else {
                return;
            };

            // Emit message_start on first real chunk.
            if state.phase == Phase::AwaitingFirstChunk {
                let data = json!({
                    "type": sse::MESSAGE_START,
                    "message": {
                        "id": format!("msg_{}", &Uuid::new_v4().simple().to_string()[..24]),
                        "type": "message",
                        "role": "assistant",
                        "model": state.original_model,
                        "content": [],
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0
                        }
                    }
                });
                buf.push_back(make_sse(sse::MESSAGE_START, &data));
                state.phase = Phase::Streaming;
                debug!("stream started, upstream id={}", msg_id);
            }

            // ── Text content delta ──────────────────────────────────────
            if let Some(ref text) = choice.delta.content {
                if !text.is_empty() {
                    // Lazily open a text block if we haven't yet.
                    if state.block_index == 0 && !state.has_tool_call && buf.is_empty() {
                        // Check if a content_block_start was already emitted
                        // for block 0. We use a simple heuristic: if block_index
                        // is 0, it's the first content — emit block_start.
                        // Actually emit it only once per block.
                    }
                    // Emit content_block_start if this is the very first text
                    // delta for a new block_index.
                    ensure_text_block_started(state, buf);

                    let data = json!({
                        "type": sse::CONTENT_BLOCK_DELTA,
                        "index": state.block_index,
                        "delta": {
                            "type": sse::DELTA_TEXT,
                            "text": text
                        }
                    });
                    buf.push_back(make_sse(sse::CONTENT_BLOCK_DELTA, &data));
                }
            }

            // ── Tool call deltas ────────────────────────────────────────
            if let Some(ref tool_calls) = choice.delta.tool_calls {
                for tc in tool_calls {
                    let acc =
                        state
                            .tool_calls
                            .entry(tc.index)
                            .or_insert_with(|| ToolCallAccumulator {
                                id: String::new(),
                                name: String::new(),
                                args: String::new(),
                            });
                    if let Some(ref id) = tc.id {
                        acc.id = id.clone();
                    }
                    if let Some(ref func) = tc.function {
                        if let Some(ref name) = func.name {
                            acc.name.clone_from(name);
                        }
                        if let Some(ref args) = func.arguments {
                            acc.args.push_str(args);
                        }
                    }
                }
                state.has_tool_call = true;
            }

            // ── finish_reason ───────────────────────────────────────────
            if let Some(ref reason) = choice.finish_reason {
                match reason.as_str() {
                    "tool_calls" => {
                        // Close any open text block before emitting tool blocks.
                        close_text_block_if_open(state, buf);
                        flush_tool_calls(state, buf);
                        state.final_stop_reason = stop_reason::TOOL_USE.to_string();
                    }
                    "length" => {
                        close_text_block_if_open(state, buf);
                        state.final_stop_reason = stop_reason::MAX_TOKENS.to_string();
                    }
                    _ => {
                        // "stop" or other
                        close_text_block_if_open(state, buf);
                        state.final_stop_reason = stop_reason::END_TURN.to_string();
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Text block tracking
// ---------------------------------------------------------------------------

/// Track whether we've started a text content block for the current block_index.
/// We need this because ChatCompletions doesn't have explicit block_start events.
static TEXT_BLOCK_SENTINEL: &str = "__text_block_open__";

fn ensure_text_block_started(state: &mut ConverterState<impl Stream>, buf: &mut VecDeque<Event>) {
    // We use the tool_calls map to stash a marker. If block_index 0 and no
    // tool call at this index exists, emit content_block_start.
    // Simpler approach: track with a bool.
    // Actually, let's use a dedicated field. But we didn't add one.
    // Instead: we'll just emit content_block_start if this is the first text
    // delta we've seen. Track this via a simple check: if block_index was
    // never incremented and we haven't emitted a block_start yet.
    //
    // The safest approach: emit content_block_start for every block_index
    // the first time we see it. We track "last started block" in the state.
    // For now, minimal approach: emit block_start at index 0 once.
    //
    // We can check if the first event in the buffer is a block_start.
    // Actually — just track it with a simple flag. We already have block_index.
    // If block_index == 0 and we need to start it:
    if !state.has_tool_call {
        if let std::collections::hash_map::Entry::Vacant(e) = state.tool_calls.entry(usize::MAX) {
            e.insert(ToolCallAccumulator {
                id: TEXT_BLOCK_SENTINEL.into(),
                name: String::new(),
                args: String::new(),
            });
            let data = json!({
                "type": sse::CONTENT_BLOCK_START,
                "index": state.block_index,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            });
            buf.push_back(make_sse(sse::CONTENT_BLOCK_START, &data));
        }
    }
}

fn close_text_block_if_open(state: &mut ConverterState<impl Stream>, buf: &mut VecDeque<Event>) {
    if state.tool_calls.remove(&usize::MAX).is_some() {
        emit_block_stop(state.block_index, buf);
        state.block_index += 1;
    }
}

// ---------------------------------------------------------------------------
// Tool call flush (I2 strategy: emit once with full repaired JSON)
// ---------------------------------------------------------------------------

fn flush_tool_calls(state: &mut ConverterState<impl Stream>, buf: &mut VecDeque<Event>) {
    if state.tool_calls.is_empty() {
        return;
    }

    // Sort by index to emit in order.
    let mut indices: Vec<usize> = state
        .tool_calls
        .keys()
        .filter(|&&k| k != usize::MAX)
        .copied()
        .collect();
    indices.sort();

    for idx in indices {
        if let Some(acc) = state.tool_calls.remove(&idx) {
            let safe_id = tool_id::sanitize(&acc.id);
            let restored_name = tool_name::restore(&state.tool_name_map, &acc.name);

            // content_block_start (tool_use)
            let start_data = json!({
                "type": sse::CONTENT_BLOCK_START,
                "index": state.block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": safe_id,
                    "name": restored_name,
                    "input": {}
                }
            });
            buf.push_back(make_sse(sse::CONTENT_BLOCK_START, &start_data));

            // Empty delta (CPA pattern)
            let empty_delta = json!({
                "type": sse::CONTENT_BLOCK_DELTA,
                "index": state.block_index,
                "delta": {
                    "type": sse::DELTA_INPUT_JSON,
                    "partial_json": ""
                }
            });
            buf.push_back(make_sse(sse::CONTENT_BLOCK_DELTA, &empty_delta));

            // Full repaired args as one delta
            let repaired = fix_json::fix_json(&acc.args);
            let args_delta = json!({
                "type": sse::CONTENT_BLOCK_DELTA,
                "index": state.block_index,
                "delta": {
                    "type": sse::DELTA_INPUT_JSON,
                    "partial_json": repaired
                }
            });
            buf.push_back(make_sse(sse::CONTENT_BLOCK_DELTA, &args_delta));

            // content_block_stop
            emit_block_stop(state.block_index, buf);
            state.block_index += 1;
        }
    }

    // Remove any leftover sentinel.
    state.tool_calls.remove(&usize::MAX);
}

// ---------------------------------------------------------------------------
// Epilogue
// ---------------------------------------------------------------------------

fn emit_epilogue(state: &ConverterState<impl Stream>, buf: &mut VecDeque<Event>) {
    let final_stop = if state.has_tool_call {
        stop_reason::TOOL_USE.to_string()
    } else {
        state.final_stop_reason.clone()
    };

    // C1 cache deduction formula.
    let cached = state.usage.cache_read_input_tokens.unwrap_or(0);
    let fresh = state.usage.input_tokens.saturating_sub(cached);
    let report_input = if state.estimated_input_tokens > 0 && fresh > 0 {
        state.estimated_input_tokens.min(fresh)
    } else if state.estimated_input_tokens > 0 {
        state.estimated_input_tokens
    } else {
        fresh
    };
    let report_output = state.usage.output_tokens;

    let usage_data = if cached > 0 {
        json!({
            "input_tokens": report_input,
            "output_tokens": report_output,
            "cache_read_input_tokens": cached
        })
    } else {
        json!({
            "input_tokens": report_input,
            "output_tokens": report_output
        })
    };

    warn!(
        "← stream done | input_tokens={} output_tokens={} cache_read={} stop={}",
        report_input, report_output, cached, final_stop,
    );

    let message_delta = json!({
        "type": sse::MESSAGE_DELTA,
        "delta": {
            "stop_reason": final_stop,
            "stop_sequence": null
        },
        "usage": usage_data
    });
    buf.push_back(make_sse(sse::MESSAGE_DELTA, &message_delta));

    let message_stop = json!({ "type": sse::MESSAGE_STOP });
    buf.push_back(make_sse(sse::MESSAGE_STOP, &message_stop));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_sse(event_name: &str, data: &serde_json::Value) -> Event {
    let json_str = serde_json::to_string(data).unwrap_or_else(|e| {
        error!("failed to serialize SSE data: {e}");
        format!(
            r#"{{"type":"error","error":{{"type":"serialization_error","message":"{}"}}}}"#,
            e
        )
    });
    Event::default().event(event_name).data(json_str)
}

fn emit_error_event(message: &str, buf: &mut VecDeque<Event>) {
    let data = json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": format!("Streaming error: {message}")
        }
    });
    buf.push_back(make_sse("error", &data));
}

fn emit_block_stop(index: usize, buf: &mut VecDeque<Event>) {
    let data = json!({
        "type": sse::CONTENT_BLOCK_STOP,
        "index": index
    });
    buf.push_back(make_sse(sse::CONTENT_BLOCK_STOP, &data));
}
