// llmapi.c -- Implementation of libllmapi
//
// A C99 library for calling LLM APIs (OpenAI-compatible).
//
// The HTTP(S) transport is injected by the caller (see llmapiSetTransport).
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

#include "llmapi.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>

#include <cjson/cJSON.h>

// ==================== Internal Constants ====================

#define LLMAPI_BUF_MAX   (100 * 1024 * 1024)  // 100 MB max per buffer
#define LLMAPI_LINE_MAX  (64 * 1024)          // 64 KB max per SSE line

// ==================== Internal Static State ====================

// ==================== Internal Dynamic Buffer ====================

typedef struct
{
    char*  data;
    size_t len;
    size_t cap;
} LlmBuf;

static void llmBufInit(LlmBuf* buf)
{
    buf->data = NULL;
    buf->len  = 0;
    buf->cap  = 0;
}

static int llmBufGrow(LlmBuf* buf, size_t needed)
{
    if (needed > LLMAPI_BUF_MAX)
    {
        return -1;
    }

    if (needed <= buf->cap)
    {
        return 0;
    }

    size_t newCap = buf->cap == 0 ? 64 : buf->cap;
    if (newCap > SIZE_MAX / 2)
    {
        return -1;
    }
    newCap *= 2;

    while (newCap < needed)
    {
        if (newCap > SIZE_MAX / 2)
        {
            return -1;
        }
        newCap *= 2;
    }

    char* p = (char*)realloc(buf->data, newCap);
    if (p == NULL)
    {
        return -1;
    }

    buf->data = p;
    buf->cap  = newCap;

    return 0;
}

static int llmBufAppend(LlmBuf* buf, const char* s)
{
    if (s == NULL)
    {
        return 0;
    }

    size_t slen = strlen(s);
    if (slen == 0)
    {
        return 0;
    }

    if (llmBufGrow(buf, buf->len + slen + 1) != 0)
    {
        return -1;
    }

    memcpy(buf->data + buf->len, s, slen + 1);
    buf->len += slen;

    return 0;
}

static int llmBufAppendFmt(LlmBuf* buf, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);

    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);

    if (needed < 0)
    {
        return -1;
    }

    if (llmBufGrow(buf, buf->len + (size_t)needed + 1) != 0)
    {
        return -1;
    }

    va_start(ap, fmt);
    vsnprintf(buf->data + buf->len, (size_t)needed + 1, fmt, ap);
    va_end(ap);

    buf->len += (size_t)needed;

    return 0;
}

static char* llmBufDetach(LlmBuf* buf)
{
    char* p = buf->data;
    llmBufInit(buf);

    return p;
}

static void llmBufFree(LlmBuf* buf)
{
    free(buf->data);
    llmBufInit(buf);
}

static void llmBufSecureFree(LlmBuf* buf)
{
    if (buf && buf->data)
    {
        memset(buf->data, 0, buf->len);
    }
    llmBufFree(buf);
}

// ==================== Internal Structures ====================

typedef struct
{
    char* role;
    char* content;
    char* toolCallsJson;  // JSON array string for assistant tool_calls
    char* toolCallId;      // for tool role messages
    char* toolCallName;    // tool function name (for display)
} LlmMsg;

struct LlmmapiReq
{
    char* model;

    int     hasMaxTokens;
    size_t  maxTokens;
    int     hasTemperature;
    double  temperature;

    LlmMsg* msgs;
    size_t  msgCount;
    size_t  msgCap;

    char**  extraKeys;
    char**  extraValues;
    size_t  extraCount;
    size_t  extraCap;
};

struct LlmmapiCtx
{
    LlmBuf errbuf;

    int               hasTransport;
    LlmmapiTransport  transport;

    int lastTotalTokens;
    int lastPromptTokens;
    int lastCompletionTokens;
};

// ==================== Internal: Validation ====================

static int llmapiHasCrlf(const char* s)
{
    if (s == NULL)
    {
        return 0;
    }

    while (*s)
    {
        if (*s == '\r' || *s == '\n')
        {
            return 1;
        }
        s++;
    }

    return 0;
}

// ==================== Internal: Message Helpers ====================

static LlmMsg llmMsgMake(const char* role, const char* content)
{
    LlmMsg msg;
    msg.role         = (char*)malloc(strlen(role) + 1);
    msg.content      = (char*)malloc(strlen(content) + 1);
    msg.toolCallsJson = NULL;
    msg.toolCallId   = NULL;
    msg.toolCallName = NULL;

    if (msg.role && msg.content)
    {
        strcpy(msg.role,    role);
        strcpy(msg.content, content);
    }
    else
    {
        free(msg.role);
        free(msg.content);
        msg.role    = NULL;
        msg.content = NULL;
    }

    return msg;
}

static void llmMsgFree(LlmMsg* msg)
{
    if (msg == NULL)
    {
        return;
    }

    free(msg->role);
    free(msg->content);
    free(msg->toolCallsJson);
    free(msg->toolCallId);
    free(msg->toolCallName);
}

// ==================== Lifecycle ====================

LlmmapiCtx* llmapiInit(void)
{
    LlmmapiCtx* ctx = (LlmmapiCtx*)malloc(sizeof(LlmmapiCtx));
    if (ctx == NULL)
    {
        return NULL;
    }

    llmBufInit(&ctx->errbuf);

    ctx->hasTransport = 0;
    memset(&ctx->transport, 0, sizeof(ctx->transport));

    ctx->lastTotalTokens      = 0;
    ctx->lastPromptTokens     = 0;
    ctx->lastCompletionTokens = 0;

    return ctx;
}

void llmapiCleanup(LlmmapiCtx* ctx)
{
    if (ctx == NULL)
    {
        return;
    }

    if (ctx->hasTransport && ctx->transport.destroy)
    {
        ctx->transport.destroy(ctx->transport.userdata);
    }

    llmBufFree(&ctx->errbuf);
    free(ctx);
}

// ==================== Error ====================

const char* llmapiStrError(LlmmapiCtx* ctx)
{
    if (ctx == NULL)
    {
        return "NULL context";
    }

    return ctx->errbuf.data ? ctx->errbuf.data : "";
}

// ==================== Transport ====================

void llmapiSetTransport(LlmmapiCtx* ctx, const LlmmapiTransport* transport)
{
    if (ctx == NULL)
    {
        return;
    }

    if (transport == NULL || transport->perform == NULL)
    {
        ctx->hasTransport = 0;
        memset(&ctx->transport, 0, sizeof(ctx->transport));
        return;
    }

    ctx->transport    = *transport;
    ctx->hasTransport = 1;
}

// ==================== Request ====================

LlmmapiReq* llmapiReqNew(const char* model)
{
    if (model == NULL)
    {
        return NULL;
    }

    LlmmapiReq* req = (LlmmapiReq*)malloc(sizeof(LlmmapiReq));
    if (req == NULL)
    {
        return NULL;
    }

    req->model = (char*)malloc(strlen(model) + 1);
    if (req->model == NULL)
    {
        free(req);
        return NULL;
    }

    strcpy(req->model, model);

    req->hasMaxTokens   = 0;
    req->maxTokens      = 0;
    req->hasTemperature = 0;
    req->temperature    = 0.0;

    req->msgs     = NULL;
    req->msgCount = 0;
    req->msgCap   = 0;

    req->extraKeys   = NULL;
    req->extraValues = NULL;
    req->extraCount  = 0;
    req->extraCap    = 0;

    return req;
}

void llmapiReqFree(LlmmapiReq* req)
{
    if (req == NULL)
    {
        return;
    }

    free(req->model);

    for (size_t i = 0; i < req->msgCount; i++)
    {
        llmMsgFree(&req->msgs[i]);
    }
    free(req->msgs);

    for (size_t i = 0; i < req->extraCount; i++)
    {
        free(req->extraKeys[i]);
        free(req->extraValues[i]);
    }
    free(req->extraKeys);
    free(req->extraValues);

    free(req);
}

void llmapiReqAddMsg(LlmmapiReq* req, const char* role, const char* content)
{
    if (req == NULL || role == NULL || content == NULL)
    {
        return;
    }

    if (llmapiHasCrlf(role))
    {
        return;
    }

    if (req->msgCount >= req->msgCap)
    {
        size_t newCap = req->msgCap == 0 ? 4 : req->msgCap * 2;
        LlmMsg* p = (LlmMsg*)realloc(req->msgs,
                                     newCap * sizeof(LlmMsg));
        if (p == NULL)
        {
            return;
        }

        req->msgs   = p;
        req->msgCap = newCap;
    }

    LlmMsg msg = llmMsgMake(role, content);
    if (msg.role == NULL || msg.content == NULL)
    {
        llmMsgFree(&msg);
        return;
    }

    req->msgs[req->msgCount++] = msg;
}

void llmapiReqAddAssistantToolCalls(LlmmapiReq* req, const char* toolCallsJson)
{
    if (req == NULL || toolCallsJson == NULL)
    {
        return;
    }

    if (req->msgCount >= req->msgCap)
    {
        size_t newCap = req->msgCap == 0 ? 4 : req->msgCap * 2;
        LlmMsg* p = (LlmMsg*)realloc(req->msgs, newCap * sizeof(LlmMsg));
        if (p == NULL)
        {
            return;
        }
        req->msgs   = p;
        req->msgCap = newCap;
    }

    LlmMsg msg;
    msg.role         = (char*)malloc(strlen("assistant") + 1);
    msg.content      = (char*)malloc(1);
    msg.toolCallsJson = (char*)malloc(strlen(toolCallsJson) + 1);
    msg.toolCallId   = NULL;
    msg.toolCallName = NULL;

    if (msg.role && msg.content && msg.toolCallsJson)
    {
        strcpy(msg.role, "assistant");
        msg.content[0] = '\0';
        strcpy(msg.toolCallsJson, toolCallsJson);
    }
    else
    {
        llmMsgFree(&msg);
        return;
    }

    req->msgs[req->msgCount++] = msg;
}

void llmapiReqAddToolResult(LlmmapiReq* req, const char* toolCallId, const char* name, const char* content)
{
    if (req == NULL || toolCallId == NULL || content == NULL)
    {
        return;
    }

    if (req->msgCount >= req->msgCap)
    {
        size_t newCap = req->msgCap == 0 ? 4 : req->msgCap * 2;
        LlmMsg* p = (LlmMsg*)realloc(req->msgs, newCap * sizeof(LlmMsg));
        if (p == NULL)
        {
            return;
        }
        req->msgs   = p;
        req->msgCap = newCap;
    }

    LlmMsg msg;
    msg.role         = (char*)malloc(strlen("tool") + 1);
    msg.content      = (char*)malloc(strlen(content) + 1);
    msg.toolCallId   = (char*)malloc(strlen(toolCallId) + 1);
    msg.toolCallsJson = NULL;
    msg.toolCallName = name ? (char*)malloc(strlen(name) + 1) : NULL;

    if (msg.role && msg.content && msg.toolCallId)
    {
        strcpy(msg.role, "tool");
        strcpy(msg.content, content);
        strcpy(msg.toolCallId, toolCallId);
        if (msg.toolCallName)
        {
            strcpy(msg.toolCallName, name);
        }
    }
    else
    {
        llmMsgFree(&msg);
        return;
    }

    req->msgs[req->msgCount++] = msg;
}

void llmapiReqSetMaxTokens(LlmmapiReq* req, size_t maxTokens)
{
    if (req == NULL)
    {
        return;
    }

    req->hasMaxTokens = 1;
    req->maxTokens    = maxTokens;
}

void llmapiReqSetTemperature(LlmmapiReq* req, double temp)
{
    if (req == NULL)
    {
        return;
    }

    req->hasTemperature = 1;
    req->temperature    = temp;
}

void llmapiReqSet(LlmmapiReq* req, const char* key, const char* jsonValue)
{
    if (req == NULL || key == NULL || jsonValue == NULL)
    {
        return;
    }

    if (req->extraCount >= req->extraCap)
    {
        size_t newCap = req->extraCap == 0 ? 4 : req->extraCap * 2;

        char** newKeys = (char**)malloc(newCap * sizeof(char*));
        if (newKeys == NULL)
        {
            return;
        }

        char** newValues = (char**)malloc(newCap * sizeof(char*));
        if (newValues == NULL)
        {
            free(newKeys);
            return;
        }

        for (size_t i = 0; i < req->extraCount; i++)
        {
            newKeys[i]   = req->extraKeys[i];
            newValues[i] = req->extraValues[i];
        }

        free(req->extraKeys);
        free(req->extraValues);
        req->extraKeys   = newKeys;
        req->extraValues = newValues;
        req->extraCap    = newCap;
    }

    req->extraKeys[req->extraCount]   = (char*)malloc(strlen(key) + 1);
    req->extraValues[req->extraCount] = (char*)malloc(strlen(jsonValue) + 1);

    if (req->extraKeys[req->extraCount] == NULL ||
        req->extraValues[req->extraCount] == NULL)
    {
        free(req->extraKeys[req->extraCount]);
        free(req->extraValues[req->extraCount]);
        return;
    }

    strcpy(req->extraKeys[req->extraCount],   key);
    strcpy(req->extraValues[req->extraCount], jsonValue);
    req->extraCount++;
}

// ==================== Internal: JSON Builder ====================

static int llmapiBuildRequestBody(const LlmmapiReq* req, int stream, LlmBuf* out)
{
    cJSON* root = cJSON_CreateObject();
    if (root == NULL)
    {
        return -1;
    }

    cJSON_AddStringToObject(root, "model", req->model);

    cJSON* msgs = cJSON_AddArrayToObject(root, "messages");
    if (msgs == NULL)
    {
        cJSON_Delete(root);
        return -1;
    }

    for (size_t i = 0; i < req->msgCount; i++)
    {
        const LlmMsg* msg = &req->msgs[i];

        cJSON* item = cJSON_CreateObject();
        if (item == NULL)
        {
            cJSON_Delete(root);
            return -1;
        }
        cJSON_AddStringToObject(item, "role", msg->role);

        if (msg->toolCallsJson)
        {
            cJSON* tcs = cJSON_Parse(msg->toolCallsJson);
            if (tcs)
            {
                cJSON_AddItemToObject(item, "tool_calls", tcs);
                cJSON_AddNullToObject(item, "content");
            }
            else
            {
                cJSON_AddStringToObject(item, "content", "");
            }
        }
        else if (msg->toolCallId)
        {
            cJSON_AddStringToObject(item, "tool_call_id", msg->toolCallId);
            if (msg->toolCallName)
            {
                cJSON_AddStringToObject(item, "name", msg->toolCallName);
            }
            cJSON_AddStringToObject(item, "content", msg->content);
        }
        else
        {
            cJSON_AddStringToObject(item, "content", msg->content);
        }

        cJSON_AddItemToArray(msgs, item);
    }

    cJSON_AddBoolToObject(root, "stream", stream ? 1 : 0);

    if (req->hasMaxTokens)
    {
        cJSON_AddNumberToObject(root, "max_tokens", (double)req->maxTokens);
    }

    if (req->hasTemperature)
    {
        cJSON_AddNumberToObject(root, "temperature", req->temperature);
    }

    for (size_t i = 0; i < req->extraCount; i++)
    {
        cJSON* val = cJSON_Parse(req->extraValues[i]);
        if (val == NULL)
        {
            val = cJSON_CreateString(req->extraValues[i]);
        }

        if (val != NULL)
        {
            cJSON_AddItemToObject(root, req->extraKeys[i], val);
        }
    }

    char* jsonStr = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (jsonStr == NULL)
    {
        return -1;
    }

    int ok = llmBufAppend(out, jsonStr);
    free(jsonStr);

    return ok;
}

// ==================== Internal: Chunk Collector ====================

// Append every received chunk to a buffer (used by the sync path).
static int llmCollectChunk(const void* data, size_t len, void* userdata)
{
    LlmBuf* buf = (LlmBuf*)userdata;

    if (llmBufGrow(buf, buf->len + len + 1) != 0)
    {
        return 0;
    }

    if (len > 0)
    {
        memcpy(buf->data + buf->len, data, len);
        buf->len += len;
    }
    buf->data[buf->len] = '\0';

    return 1;
}

// ==================== Internal: Request Assembly ====================

// Fill urlBuf, authBuf and the NULL-terminated header array.  `headers`
// must have room for at least 4 entries.  Returns 0 on success, -1 on
// failure (with a message appended to ctx->errbuf).
static int llmapiBuildRequest(LlmmapiCtx* ctx,
                               const char* baseUrl,
                               const char* apiKey,
                               int stream,
                               LlmBuf* urlBuf,
                               LlmBuf* authBuf,
                               const char** headers)
{
    if (llmapiHasCrlf(baseUrl))
    {
        llmBufAppend(&ctx->errbuf, "baseUrl contains invalid characters");
        return -1;
    }

    if (llmapiHasCrlf(apiKey))
    {
        llmBufAppend(&ctx->errbuf, "apiKey contains invalid characters");
        return -1;
    }

    llmBufAppend(urlBuf, baseUrl);
    size_t baseLen = urlBuf->len;
    if (baseLen > 0 && urlBuf->data[baseLen - 1] != '/')
    {
        llmBufAppend(urlBuf, "/chat/completions");
    }
    else
    {
        llmBufAppend(urlBuf, "chat/completions");
    }

    llmBufAppend(authBuf, "Authorization: Bearer ");
    llmBufAppend(authBuf, apiKey);

    size_t n = 0;
    headers[n++] = "Content-Type: application/json";
    if (stream)
    {
        headers[n++] = "Accept: text/event-stream";
    }
    headers[n++] = authBuf->data;
    headers[n]   = NULL;

    return 0;
}

// ==================== Sync Chat ====================

char* llmapiChatComplete(LlmmapiCtx* ctx,
                          const char* baseUrl,
                          const char* apiKey,
                          const LlmmapiReq* req)
{
    if (ctx == NULL || baseUrl == NULL || apiKey == NULL || req == NULL)
    {
        return NULL;
    }

    llmBufFree(&ctx->errbuf);
    llmBufInit(&ctx->errbuf);

    if (!ctx->hasTransport)
    {
        llmBufAppend(&ctx->errbuf,
                     "no transport set (call llmapiSetTransport)");
        return NULL;
    }

    LlmBuf bodyBuf;
    llmBufInit(&bodyBuf);

    if (llmapiBuildRequestBody(req, 0, &bodyBuf) != 0)
    {
        llmBufAppend(&ctx->errbuf, "Failed to build request body");
        llmBufFree(&bodyBuf);
        return NULL;
    }

    LlmBuf urlBuf;
    LlmBuf authBuf;
    llmBufInit(&urlBuf);
    llmBufInit(&authBuf);

    const char* headers[4];
    if (llmapiBuildRequest(ctx, baseUrl, apiKey, 0,
                           &urlBuf, &authBuf, headers) != 0)
    {
        llmBufFree(&bodyBuf);
        llmBufFree(&urlBuf);
        llmBufSecureFree(&authBuf);
        return NULL;
    }

    LlmmapiHttpRequest httpReq;
    httpReq.method    = "POST";
    httpReq.url       = urlBuf.data;
    httpReq.headers   = headers;
    httpReq.body      = bodyBuf.data;
    httpReq.bodyLen   = bodyBuf.len;
    httpReq.timeoutMs = 120000L;

    LlmBuf respBuf;
    llmBufInit(&respBuf);

    LlmmapiHttpResult result;
    result.status = 0;
    result.error  = NULL;

    int rc = ctx->transport.perform(ctx->transport.userdata, &httpReq,
                                    llmCollectChunk, &respBuf, &result);

    llmBufFree(&bodyBuf);
    llmBufFree(&urlBuf);
    llmBufSecureFree(&authBuf);

    if (rc != 0 || (result.error != NULL && result.error[0] != '\0'))
    {
        llmBufAppend(&ctx->errbuf,
                     (result.error != NULL) ? result.error
                                            : "transport error");
        llmBufFree(&respBuf);
        return NULL;
    }

    if (result.status < 200 || result.status >= 300)
    {
        llmBufAppendFmt(&ctx->errbuf,
                        "HTTP %ld: %s",
                        result.status,
                        respBuf.data ? respBuf.data : "");
        llmBufFree(&respBuf);
        return NULL;
    }

    return llmBufDetach(&respBuf);
}

// ==================== Internal: Streaming SSE Parser ====================

struct LlmStreamCtx
{
    LlmBuf           lineBuf;
    LlmBuf           rawBuf;
    LlmmapiStreamCb  cb;
    void*            userdata;
    int              done;
    int              usageTotal;
    int              usagePrompt;
    int              usageCompletion;
};

static void llmProcessSseLine(struct LlmStreamCtx* sctx, const char* line,
                               size_t len)
{
    if (sctx->done || len == 0)
    {
        return;
    }

    if (len < 6 || strncmp(line, "data: ", 6) != 0)
    {
        return;
    }

    const char* payload = line + 6;
    size_t payloadLen = len - 6;

    if (payloadLen == 6 && memcmp(payload, "[DONE]", 6) == 0)
    {
        sctx->done = 1;
        sctx->cb(LLMAPI_STREAM_TEXT, NULL, 1, sctx->userdata);
        return;
    }

    char* jsonStr = (char*)malloc(payloadLen + 1);
    if (jsonStr == NULL)
    {
        return;
    }

    memcpy(jsonStr, payload, payloadLen);
    jsonStr[payloadLen] = '\0';

    cJSON* root = cJSON_Parse(jsonStr);
    free(jsonStr);

    if (root == NULL)
    {
        return;
    }

    cJSON* choices = cJSON_GetObjectItem(root, "choices");
    if (choices != NULL && cJSON_IsArray(choices) &&
        cJSON_GetArraySize(choices) > 0)
    {
        cJSON* choice = cJSON_GetArrayItem(choices, 0);
        cJSON* delta  = cJSON_GetObjectItem(choice, "delta");

        if (delta != NULL)
        {
            // standard text content
            cJSON* content = cJSON_GetObjectItem(delta, "content");
            if (content != NULL && cJSON_IsString(content) &&
                content->valuestring[0] != '\0')
            {
                sctx->cb(LLMAPI_STREAM_TEXT,
                         content->valuestring, 0, sctx->userdata);
            }

            // reasoning / thinking content
            cJSON* reasoning = cJSON_GetObjectItem(delta, "reasoning_content");
            if (reasoning != NULL && cJSON_IsString(reasoning) &&
                reasoning->valuestring[0] != '\0')
            {
                sctx->cb(LLMAPI_STREAM_THINKING,
                         reasoning->valuestring, 0, sctx->userdata);
            }

            // tool calls
            cJSON* toolCalls = cJSON_GetObjectItem(delta, "tool_calls");
            if (toolCalls != NULL && cJSON_IsArray(toolCalls) &&
                cJSON_GetArraySize(toolCalls) > 0)
            {
                char* tcJson = cJSON_PrintUnformatted(toolCalls);
                if (tcJson)
                {
                    sctx->cb(LLMAPI_STREAM_TOOL_CALL,
                             tcJson, 0, sctx->userdata);
                    free(tcJson);
                }
            }
        }

        cJSON* finish = cJSON_GetObjectItem(choice, "finish_reason");
        if (finish != NULL && cJSON_IsString(finish) &&
            finish->valuestring[0] != '\0')
        {
            sctx->done = 1;
            sctx->cb(LLMAPI_STREAM_TEXT, NULL, 1, sctx->userdata);
        }
    }

    // Some providers send token usage in the final SSE chunk.  Capture it so
    // callers can report context usage after a streaming call completes.
    cJSON* usage = cJSON_GetObjectItem(root, "usage");
    if (usage != NULL && cJSON_IsObject(usage))
    {
        cJSON* item = cJSON_GetObjectItem(usage, "total_tokens");
        if (item != NULL && cJSON_IsNumber(item))
        {
            sctx->usageTotal = (int)item->valuedouble;
        }
        item = cJSON_GetObjectItem(usage, "prompt_tokens");
        if (item != NULL && cJSON_IsNumber(item))
        {
            sctx->usagePrompt = (int)item->valuedouble;
        }
        item = cJSON_GetObjectItem(usage, "completion_tokens");
        if (item != NULL && cJSON_IsNumber(item))
        {
            sctx->usageCompletion = (int)item->valuedouble;
        }
    }

    cJSON_Delete(root);
}

static int llmStreamWriteCallback(const void* data,
                                   size_t total,
                                   void* userdata)
{
    struct LlmStreamCtx* sctx = (struct LlmStreamCtx*)userdata;
    const char* ptr = (const char*)data;

    if (llmBufGrow(&sctx->rawBuf, sctx->rawBuf.len + total + 1) != 0)
    {
        return 0;
    }

    memcpy(sctx->rawBuf.data + sctx->rawBuf.len, ptr, total);
    sctx->rawBuf.len += total;
    sctx->rawBuf.data[sctx->rawBuf.len] = '\0';

    for (size_t i = 0; i < total; i++)
    {
        char c = ptr[i];

        if (c == '\n')
        {
            llmProcessSseLine(sctx,
                              sctx->lineBuf.data,
                              sctx->lineBuf.len);

            sctx->lineBuf.len = 0;
            if (sctx->lineBuf.data)
            {
                sctx->lineBuf.data[0] = '\0';
            }
        }
        else if (c == '\r')
        {
            // skip
        }
        else
        {
            if (sctx->lineBuf.len >= LLMAPI_LINE_MAX)
            {
                return 0;
            }

            if (llmBufGrow(&sctx->lineBuf, sctx->lineBuf.len + 2) != 0)
            {
                return 0;
            }

            sctx->lineBuf.data[sctx->lineBuf.len++] = c;
            sctx->lineBuf.data[sctx->lineBuf.len]   = '\0';
        }
    }

    return 1;
}

// ==================== Streaming Chat ====================

int llmapiChatCompleteStream(LlmmapiCtx* ctx,
                              const char* baseUrl,
                              const char* apiKey,
                              const LlmmapiReq* req,
                              LlmmapiStreamCb cb,
                              void* userdata)
{
    if (ctx == NULL || baseUrl == NULL || apiKey == NULL ||
        req == NULL || cb == NULL)
    {
        return -1;
    }

    llmBufFree(&ctx->errbuf);
    llmBufInit(&ctx->errbuf);

    if (!ctx->hasTransport)
    {
        llmBufAppend(&ctx->errbuf,
                     "no transport set (call llmapiSetTransport)");
        return -1;
    }

    LlmBuf bodyBuf;
    llmBufInit(&bodyBuf);

    if (llmapiBuildRequestBody(req, 1, &bodyBuf) != 0)
    {
        llmBufAppend(&ctx->errbuf, "Failed to build request body");
        llmBufFree(&bodyBuf);
        return -1;
    }

    LlmBuf urlBuf;
    LlmBuf authBuf;
    llmBufInit(&urlBuf);
    llmBufInit(&authBuf);

    const char* headers[4];
    if (llmapiBuildRequest(ctx, baseUrl, apiKey, 1,
                           &urlBuf, &authBuf, headers) != 0)
    {
        llmBufFree(&bodyBuf);
        llmBufFree(&urlBuf);
        llmBufSecureFree(&authBuf);
        return -1;
    }

    struct LlmStreamCtx sctx;
    llmBufInit(&sctx.lineBuf);
    llmBufInit(&sctx.rawBuf);
    sctx.cb       = cb;
    sctx.userdata = userdata;
    sctx.done     = 0;
    sctx.usageTotal      = 0;
    sctx.usagePrompt     = 0;
    sctx.usageCompletion = 0;

    LlmmapiHttpRequest httpReq;
    httpReq.method    = "POST";
    httpReq.url       = urlBuf.data;
    httpReq.headers   = headers;
    httpReq.body      = bodyBuf.data;
    httpReq.bodyLen   = bodyBuf.len;
    httpReq.timeoutMs = 120000L;

    LlmmapiHttpResult result;
    result.status = 0;
    result.error  = NULL;

    int rc = ctx->transport.perform(ctx->transport.userdata, &httpReq,
                                    llmStreamWriteCallback, &sctx, &result);

    llmBufFree(&bodyBuf);
    llmBufFree(&urlBuf);
    llmBufSecureFree(&authBuf);

    if (sctx.lineBuf.len > 0)
    {
        llmProcessSseLine(&sctx, sctx.lineBuf.data, sctx.lineBuf.len);
    }
    llmBufFree(&sctx.lineBuf);

    ctx->lastTotalTokens      = sctx.usageTotal;
    ctx->lastPromptTokens     = sctx.usagePrompt;
    ctx->lastCompletionTokens = sctx.usageCompletion;

    if (rc != 0 || (result.error != NULL && result.error[0] != '\0'))
    {
        llmBufAppend(&ctx->errbuf,
                     (result.error != NULL) ? result.error
                                            : "transport error");
        llmBufFree(&sctx.rawBuf);
        return -1;
    }

    if (result.status < 200 || result.status >= 300)
    {
        llmBufAppendFmt(&ctx->errbuf,
                        "HTTP %ld: %s",
                        result.status,
                        sctx.rawBuf.data ? sctx.rawBuf.data : "");
        llmBufFree(&sctx.rawBuf);
        return -1;
    }

    llmBufFree(&sctx.rawBuf);

    if (!sctx.done)
    {
        sctx.cb(LLMAPI_STREAM_TEXT, NULL, 1, sctx.userdata);
    }

    return 0;
}

// Retrieve the token usage captured from the last streaming response.
// Fields are set to 0 when no usage was reported by the provider.
void llmapiStreamGetUsage(LlmmapiCtx* ctx,
                          int* total,
                          int* prompt,
                          int* completion)
{
    if (ctx == NULL)
    {
        if (total)      *total = 0;
        if (prompt)     *prompt = 0;
        if (completion) *completion = 0;
        return;
    }
    if (total)      *total      = ctx->lastTotalTokens;
    if (prompt)     *prompt     = ctx->lastPromptTokens;
    if (completion) *completion = ctx->lastCompletionTokens;
}

// ==================== Response Helpers ====================

char* llmapiRespGetContent(const char* json, size_t idx)
{
    if (json == NULL)
    {
        return NULL;
    }

    cJSON* root = cJSON_Parse(json);
    if (root == NULL)
    {
        return NULL;
    }

    char* result = NULL;

    cJSON* choices = cJSON_GetObjectItem(root, "choices");
    if (choices != NULL && cJSON_IsArray(choices) &&
        (int)idx < cJSON_GetArraySize(choices))
    {
        cJSON* choice  = cJSON_GetArrayItem(choices, idx);
        cJSON* message = cJSON_GetObjectItem(choice, "message");
        if (message != NULL)
        {
            cJSON* content = cJSON_GetObjectItem(message, "content");
            if (content != NULL && cJSON_IsString(content))
            {
                result = strdup(content->valuestring);
            }
        }
    }

    cJSON_Delete(root);

    return result;
}

char* llmapiRespGetReasoningContent(const char* json, size_t idx)
{
    if (json == NULL)
    {
        return NULL;
    }

    cJSON* root = cJSON_Parse(json);
    if (root == NULL)
    {
        return NULL;
    }

    char* result = NULL;

    cJSON* choices = cJSON_GetObjectItem(root, "choices");
    if (choices != NULL && cJSON_IsArray(choices) &&
        (int)idx < cJSON_GetArraySize(choices))
    {
        cJSON* choice  = cJSON_GetArrayItem(choices, idx);
        cJSON* message = cJSON_GetObjectItem(choice, "message");
        if (message != NULL)
        {
            cJSON* reasoning = cJSON_GetObjectItem(message, "reasoning_content");
            if (reasoning != NULL && cJSON_IsString(reasoning))
            {
                result = strdup(reasoning->valuestring);
            }
        }
    }

    cJSON_Delete(root);

    return result;
}

char* llmapiRespGetToolCallsJson(const char* json, size_t idx)
{
    if (json == NULL)
    {
        return NULL;
    }

    cJSON* root = cJSON_Parse(json);
    if (root == NULL)
    {
        return NULL;
    }

    char* result = NULL;

    cJSON* choices = cJSON_GetObjectItem(root, "choices");
    if (choices != NULL && cJSON_IsArray(choices) &&
        (int)idx < cJSON_GetArraySize(choices))
    {
        cJSON* choice  = cJSON_GetArrayItem(choices, idx);
        cJSON* message = cJSON_GetObjectItem(choice, "message");
        if (message != NULL)
        {
            cJSON* toolCalls = cJSON_GetObjectItem(message, "tool_calls");
            if (toolCalls != NULL && cJSON_IsArray(toolCalls) &&
                cJSON_GetArraySize(toolCalls) > 0)
            {
                result = cJSON_PrintUnformatted(toolCalls);
            }
        }
    }

    cJSON_Delete(root);

    return result;
}

int llmapiRespGetTotalTokens(const char* json)
{
    if (json == NULL) return -1;

    cJSON* root = cJSON_Parse(json);
    if (root == NULL) return -1;

    int val = -1;
    cJSON* usage = cJSON_GetObjectItem(root, "usage");
    if (usage != NULL)
    {
        cJSON* tokens = cJSON_GetObjectItem(usage, "total_tokens");
        if (tokens != NULL && cJSON_IsNumber(tokens))
        {
            val = tokens->valueint;
        }
    }

    cJSON_Delete(root);
    return val;
}

int llmapiRespGetPromptTokens(const char* json)
{
    if (json == NULL) return -1;

    cJSON* root = cJSON_Parse(json);
    if (root == NULL) return -1;

    int val = -1;
    cJSON* usage = cJSON_GetObjectItem(root, "usage");
    if (usage != NULL)
    {
        cJSON* tokens = cJSON_GetObjectItem(usage, "prompt_tokens");
        if (tokens != NULL && cJSON_IsNumber(tokens))
        {
            val = tokens->valueint;
        }
    }

    cJSON_Delete(root);
    return val;
}

int llmapiRespGetCompletionTokens(const char* json)
{
    if (json == NULL) return -1;

    cJSON* root = cJSON_Parse(json);
    if (root == NULL) return -1;

    int val = -1;
    cJSON* usage = cJSON_GetObjectItem(root, "usage");
    if (usage != NULL)
    {
        cJSON* tokens = cJSON_GetObjectItem(usage, "completion_tokens");
        if (tokens != NULL && cJSON_IsNumber(tokens))
        {
            val = tokens->valueint;
        }
    }

    cJSON_Delete(root);
    return val;
}
