// llmapi.h -- Public API for libllmapi
//
// A C99 library for calling LLM APIs (OpenAI-compatible) using libcurl and cjson.
//
// Copyright (c) 2025, libllmapi contributors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef LLMAPI_H
#define LLMAPI_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>

// ==================== Lifecycle ====================
//
// NOTE: llmapiInit() / llmapiCleanup() are not thread-safe with respect to
// each other.  Serialise all init/cleanup calls (e.g. do them in the main
// thread before spawning workers).  Once a context is created, using it from
// multiple threads concurrently IS safe.

typedef struct LlmmapiCtx LlmmapiCtx;

LlmmapiCtx* llmapiInit(void);
void        llmapiCleanup(LlmmapiCtx* ctx);

// ==================== Error ====================

const char* llmapiStrError(LlmmapiCtx* ctx);

// ==================== Configuration ====================

// Set custom CA bundle path for TLS verification.
// Pass NULL to use the default system CA path.
void llmapiSetCaPath(LlmmapiCtx* ctx, const char* caPath);

// ==================== Role Macros ====================

#define LLMAPI_ROLE_SYSTEM    "system"
#define LLMAPI_ROLE_USER      "user"
#define LLMAPI_ROLE_ASSISTANT "assistant"

// ==================== Request ====================

typedef struct LlmmapiReq LlmmapiReq;

LlmmapiReq* llmapiReqNew(const char* model);
void        llmapiReqFree(LlmmapiReq* req);

// Add a message to the request.  role is "system", "user", "assistant", or "tool".
// The role and content strings are copied internally.
void llmapiReqAddMsg(LlmmapiReq* req, const char* role, const char* content);

// Add an assistant message with tool_calls (content is null).
// toolCallsJson must be a valid JSON array of tool call objects.
void llmapiReqAddAssistantToolCalls(LlmmapiReq* req, const char* toolCallsJson);

// Add a tool result message (role="tool") with a tool_call_id.
void llmapiReqAddToolResult(LlmmapiReq* req, const char* toolCallId, const char* name, const char* content);

void llmapiReqSetMaxTokens(LlmmapiReq* req, size_t maxTokens);
void llmapiReqSetTemperature(LlmmapiReq* req, double temp);

// Generic setter for any Chat Completions parameter.
// jsonValue must be a valid JSON literal:
//   llmapiReqSet(req, "top_p", "0.9")
//   llmapiReqSet(req, "stop",   "[\"END\"]")
// If jsonValue is not parseable as JSON, it is treated as a raw string.
void llmapiReqSet(LlmmapiReq* req, const char* key, const char* jsonValue);

// ==================== Sync Chat ====================

// Returns a malloc'd JSON string (full API response) on success, or NULL.
// Caller must free the returned string.
char* llmapiChatComplete(LlmmapiCtx* ctx,
                          const char* baseUrl,
                          const char* apiKey,
                          const LlmmapiReq* req);

// ==================== Streaming ====================

// Content type for streaming callbacks.
typedef enum
{
    LLMAPI_STREAM_TEXT,       // standard text delta
    LLMAPI_STREAM_THINKING,   // reasoning / thinking content
    LLMAPI_STREAM_TOOL_CALL,  // tool call delta (JSON string)
} LlmmapiStreamType;

// Callback invoked for each streaming chunk.
// @param type     Content type (text, thinking, tool call).
// @param content  Text / reasoning / tool-call-JSON delta (may be NULL).
//                 Pointer is valid only during the callback;
//                 strdup() for persistence.
// @param done     1 if final chunk, 0 otherwise.
// @param userdata User-supplied pointer.
typedef void (*LlmmapiStreamCb)(LlmmapiStreamType type,
                                 const char* content,
                                 int done,
                                 void* userdata);

// Returns 0 on success, -1 on failure.
int llmapiChatCompleteStream(LlmmapiCtx* ctx,
                              const char* baseUrl,
                              const char* apiKey,
                              const LlmmapiReq* req,
                              LlmmapiStreamCb cb,
                              void* userdata);

// Retrieve the token usage captured from the last streaming response.
// Fields are set to 0 when the provider reported no usage.
void llmapiStreamGetUsage(LlmmapiCtx* ctx,
                          int* total,
                          int* prompt,
                          int* completion);

// ==================== Response Helpers ====================

// Extract text content from choices[idx].message.content.
// Returns malloc'd string (caller must free) or NULL.
char* llmapiRespGetContent(const char* json, size_t idx);

// Extract reasoning/thinking content from choices[idx].message.reasoning_content.
// Returns malloc'd string (caller must free) or NULL.
char* llmapiRespGetReasoningContent(const char* json, size_t idx);

// Extract serialised tool_calls JSON from choices[idx].message.tool_calls.
// Returns malloc'd JSON string or NULL.
char* llmapiRespGetToolCallsJson(const char* json, size_t idx);

int   llmapiRespGetTotalTokens(const char* json);
int   llmapiRespGetPromptTokens(const char* json);
int   llmapiRespGetCompletionTokens(const char* json);

#ifdef __cplusplus
}
#endif

#endif /* LLMAPI_H */
