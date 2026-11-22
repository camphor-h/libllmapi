// llmapi.h -- Public API for libllmapi
//
// A C99 library for calling LLM APIs.  The HTTP(S) transport is injected by
// the caller (see llmapiSetTransport), and the wire protocol is selected with
// llmapiSetApi() (OpenAI / Anthropic).  JSON is handled with cjson.
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
// NOTE: a context is NOT safe to use concurrently from multiple threads.
// It holds shared mutable state (the error buffer and the last token-usage
// counters) that is written without locking.  Serialise calls per context, or
// give each thread its own context.

typedef struct LlmmapiCtx LlmmapiCtx;

LlmmapiCtx* llmapiInit(void);
void        llmapiCleanup(LlmmapiCtx* ctx);

// ==================== Error ====================

const char* llmapiStrError(LlmmapiCtx* ctx);

// ==================== Transport ====================
//
// libllmapi does not perform any network I/O itself.  You must inject a
// transport that does the actual HTTP(S) request.  This keeps the library
// independent of any particular HTTP/TLS stack (libcurl, WinHTTP,
// esp_http_client, a raw socket, a mock, ...).

// Callback invoked by the transport for each chunk of the response body.
// Return non-zero to continue, or 0 to abort the transfer.
typedef int (*LlmmapiWriteFn)(const void* data, size_t len, void* userdata);

// A fully-formed HTTP request handed to the transport.
typedef struct
{
    const char*        method;    // always "POST"
    const char*        url;       // baseUrl + provider path
    const char* const* headers;   // NULL-terminated list
    const void*        body;      // serialized request body
    size_t             bodyLen;
    long               timeoutMs; // suggested timeout; transport may ignore
} LlmmapiHttpRequest;

// Filled in by the transport.
typedef struct
{
    long        status; // HTTP status code (required on success)
    const char* error;  // non-NULL on transport failure; may be static storage
} LlmmapiHttpResult;

typedef struct LlmmapiTransport
{
    // Perform the request.  Response body bytes are delivered through onChunk.
    // Return 0 when the round-trip completed (regardless of HTTP status),
    // or non-zero on a transport-level failure (set result->error).
    int (*perform)(void* userdata,
                   const LlmmapiHttpRequest* request,
                   LlmmapiWriteFn onChunk,
                   void* chunkUserdata,
                   LlmmapiHttpResult* result);

    // Optional.  Invoked once from llmapiCleanup() for the currently
    // injected transport.  Replacing a transport does not call this.
    void (*destroy)(void* userdata);

    void* userdata;
} LlmmapiTransport;

// Inject the transport used by this context.  The struct is copied.
// Pass NULL to detach.  Without a transport, every call fails.
void llmapiSetTransport(LlmmapiCtx* ctx, const LlmmapiTransport* transport);

// ==================== Provider (wire protocol) ====================

typedef enum
{
    LLMMAPI_API_OPENAI = 0,        // POST /chat/completions
    LLMMAPI_API_ANTHROPIC,         // POST /messages
    LLMMAPI_API_OPENAI_RESPONSES,  // POST /responses
} LlmmapiApi;

// Select the wire protocol.  Default is LLMMAPI_API_OPENAI.
void llmapiSetApi(LlmmapiCtx* ctx, LlmmapiApi api);

// ==================== Roles ====================

#define LLMAPI_ROLE_SYSTEM    "system"
#define LLMAPI_ROLE_USER      "user"
#define LLMAPI_ROLE_ASSISTANT "assistant"
#define LLMAPI_ROLE_TOOL      "tool"

// ==================== Neutral content model ====================
//
// A message is a role plus a list of typed parts.  Providers translate this
// model to/from their own wire format, so the same request works across APIs.

typedef enum
{
    LLMMAPI_PART_TEXT,        // text: plain text
    LLMMAPI_PART_IMAGE,       // url: http(s) URL or data: URL
    LLMMAPI_PART_FILE,        // filename + mimeType + (url or fileId)
    LLMMAPI_PART_THINKING,    // text: reasoning (assistant output only)
    LLMMAPI_PART_TOOL_CALL,   // toolCallId + toolName + text(arguments JSON)
    LLMMAPI_PART_TOOL_RESULT, // toolCallId + text(result content)
} LlmmapiPartType;

typedef struct
{
    LlmmapiPartType type;

    char* text;       // TEXT / THINKING / TOOL_CALL arguments / TOOL_RESULT
    char* url;        // IMAGE / FILE (http(s) or data: URL)
    char* mimeType;   // IMAGE / FILE
    char* filename;   // FILE
    char* fileId;     // FILE (provider-side file reference, if any)
    char* toolCallId; // TOOL_CALL / TOOL_RESULT
    char* toolName;   // TOOL_CALL
} LlmmapiPart;

typedef struct
{
    char*         role;
    LlmmapiPart*  parts;
    size_t        partCount;
} LlmmapiMessage;

// Number of parts in a message.
size_t llmapiMessageGetPartCount(const LlmmapiMessage* msg);

// Borrowed pointer to a part, or NULL when idx is out of range.
const LlmmapiPart* llmapiMessageGetPart(const LlmmapiMessage* msg, size_t idx);

// ==================== Request ====================

typedef struct LlmmapiReq LlmmapiReq;

LlmmapiReq* llmapiReqNew(const char* model);
void        llmapiReqFree(LlmmapiReq* req);

// Append a new message and return a handle to it (owned by req).
LlmmapiMessage* llmapiReqAddMessage(LlmmapiReq* req, const char* role);

// Convenience: add a message with a single TEXT part.
void llmapiReqAddMsg(LlmmapiReq* req, const char* role, const char* text);

// Add parts to a message.
void llmapiMsgAddText(LlmmapiMessage* msg, const char* text);

// url may be an http(s) URL or a data: URL (caller does any base64 encoding).
void llmapiMsgAddImage(LlmmapiMessage* msg, const char* url);

// dataUrlOrFileId: if it starts with "data:" it is treated as an inline data
// URL, otherwise as a provider-side file id.
void llmapiMsgAddFile(LlmmapiMessage* msg, const char* filename,
                      const char* mimeType, const char* dataUrlOrFileId);

// argsJson is a JSON object string with the tool arguments.
void llmapiMsgAddToolCall(LlmmapiMessage* msg, const char* toolCallId,
                          const char* toolName, const char* argsJson);

// Use on a message whose role is LLMAPI_ROLE_TOOL.
void llmapiMsgAddToolResult(LlmmapiMessage* msg, const char* toolCallId,
                            const char* content);

// Register a tool. parametersJson is a JSON Schema object string.
void llmapiReqAddTool(LlmmapiReq* req, const char* name,
                      const char* description, const char* parametersJson);

void llmapiReqSetMaxTokens(LlmmapiReq* req, size_t maxTokens);
void llmapiReqSetTemperature(LlmmapiReq* req, double temp);
void llmapiReqSetTopP(LlmmapiReq* req, double topP);

// jsonArray is a JSON array string, e.g. "[\"END\"]".
void llmapiReqSetStop(LlmmapiReq* req, const char* jsonArray);

// Escape hatch: set a provider-native top-level parameter.
void llmapiReqSetExtra(LlmmapiReq* req, const char* key, const char* jsonValue);

// ==================== Response ====================

typedef struct LlmmapiResp LlmmapiResp;

// Borrowed assistant message (role="assistant", parts filled in).
const LlmmapiMessage* llmapiRespGetMessage(const LlmmapiResp* resp);

// Concatenated TEXT parts.  Returns a malloc'd string, or NULL if none.
char* llmapiRespGetText(const LlmmapiResp* resp);

// Concatenated THINKING parts.  Returns a malloc'd string, or NULL if none.
char* llmapiRespGetThinking(const LlmmapiResp* resp);

// Borrowed finish/stop reason string, or NULL.
const char* llmapiRespGetFinishReason(const LlmmapiResp* resp);

// Token usage.  Any output pointer may be NULL.
void llmapiRespGetUsage(const LlmmapiResp* resp,
                        long* input, long* output, long* total);

// Borrowed raw provider response body, or NULL.
const char* llmapiRespGetRawJson(const LlmmapiResp* resp);

void llmapiRespFree(LlmmapiResp* resp);

// ==================== Completion ====================

LlmmapiResp* llmapiChatComplete(LlmmapiCtx* ctx,
                                const char* baseUrl,
                                const char* apiKey,
                                const LlmmapiReq* req);

// Neutral streaming events.
typedef enum
{
    LLMAPI_STREAM_TEXT,       // text delta
    LLMAPI_STREAM_THINKING,   // reasoning delta
    LLMAPI_STREAM_TOOL_CALL,  // tool name/id (when known) and arguments delta
    LLMAPI_STREAM_DONE,       // final event
} LlmmapiStreamType;

typedef struct
{
    LlmmapiStreamType type;
    const char*       text;       // TEXT / THINKING / partial tool arguments
    const char*       toolCallId; // TOOL_CALL (may be NULL on later deltas)
    const char*       toolName;   // TOOL_CALL (may be NULL on later deltas)
    size_t            toolIndex;  // TOOL_CALL index
} LlmmapiStreamEvent;

typedef void (*LlmmapiStreamCb)(const LlmmapiStreamEvent* ev, void* userdata);

// Returns 0 on success, -1 on failure.
int llmapiChatCompleteStream(LlmmapiCtx* ctx,
                             const char* baseUrl,
                             const char* apiKey,
                             const LlmmapiReq* req,
                             LlmmapiStreamCb cb,
                             void* userdata);

// Token usage captured from the last streaming call.
void llmapiStreamGetUsage(LlmmapiCtx* ctx,
                          long* total, long* input, long* output);

// ==================== Raw Request ====================

// Perform an arbitrary HTTP request through the injected transport and return
// the raw response body (malloc'd; caller frees) on HTTP 2xx, or NULL.
//
// - method:       "GET", "POST", "PUT", "DELETE", ...
// - path:         appended to baseUrl, e.g. "files" or "/files"
// - contentType:  value for Content-Type, or NULL to omit it
// - apiKey:       if non-NULL, adds an Authorization header using the selected
//                 provider's scheme
// - body/bodyLen: raw request bytes (NULL/0 for GET)
char* llmapiRequest(LlmmapiCtx* ctx,
                    const char* baseUrl,
                    const char* apiKey,
                    const char* method,
                    const char* path,
                    const char* contentType,
                    const void* body,
                    size_t bodyLen);

#ifdef __cplusplus
}
#endif

#endif /* LLMAPI_H */
