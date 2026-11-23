// llmapi.c -- Implementation of libllmapi
//
// Neutral message model + pluggable wire providers (OpenAI / Anthropic).
// HTTP(S) transport is injected by the caller.
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

#define LLMAPI_BUF_MAX            (100 * 1024 * 1024)
#define LLMAPI_LINE_MAX           (64 * 1024)
#define LLMAPI_DEFAULT_MAX_TOKENS 4096
#define LLMAPI_ANTHROPIC_VERSION  "2023-06-01"

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
    char* name;
    char* description;
    char* parametersJson;
} LlmTool;

struct LlmmapiReq
{
    char* model;

    LlmmapiMessage** messages;
    size_t           messageCount;
    size_t           messageCap;

    LlmTool* tools;
    size_t   toolCount;
    size_t   toolCap;

    char** rawTools;   // provider-native tool objects (JSON object strings)
    size_t rawToolCount;
    size_t rawToolCap;

    int    hasMaxTokens;
    size_t maxTokens;

    int    hasTemperature;
    double temperature;

    int    hasTopP;
    double topP;

    char* stopJson;

    char** extraKeys;
    char** extraValues;
    size_t extraCount;
    size_t extraCap;
};

struct LlmmapiCtx
{
    LlmBuf errbuf;

    int              hasTransport;
    LlmmapiTransport transport;

    LlmmapiApi api;

    long lastTotalTokens;
    long lastPromptTokens;
    long lastCompletionTokens;
};

struct LlmmapiResp
{
    LlmmapiMessage message;
    char*          finishReason;
    long           inputTokens;
    long           outputTokens;
    long           totalTokens;
    char*          rawJson;

    LlmmapiCitation* citations;
    size_t           citationCount;
    size_t           citationCap;
};

// ==================== Internal: String helpers ====================

static char* llmStrDup(const char* s)
{
    if (s == NULL)
    {
        return NULL;
    }

    size_t n = strlen(s) + 1;
    char* p = (char*)malloc(n);
    if (p != NULL)
    {
        memcpy(p, s, n);
    }

    return p;
}

static char* llmStrNDup(const char* s, size_t n)
{
    char* p = (char*)malloc(n + 1);
    if (p == NULL)
    {
        return NULL;
    }

    memcpy(p, s, n);
    p[n] = '\0';

    return p;
}

static int llmHasCrlf(const char* s)
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

// ==================== Internal: Part / message helpers ====================

static void llmPartFree(LlmmapiPart* p)
{
    if (p == NULL)
    {
        return;
    }

    free(p->text);
    free(p->url);
    free(p->mimeType);
    free(p->filename);
    free(p->fileId);
    free(p->toolCallId);
    free(p->toolName);
    free(p->provider);
    free(p->rawJson);

    memset(p, 0, sizeof(*p));
}

static void llmMessageInit(LlmmapiMessage* msg)
{
    msg->role      = NULL;
    msg->parts     = NULL;
    msg->partCount = 0;
}

static void llmMessageFree(LlmmapiMessage* msg)
{
    if (msg == NULL)
    {
        return;
    }

    free(msg->role);

    for (size_t i = 0; i < msg->partCount; i++)
    {
        llmPartFree(&msg->parts[i]);
    }
    free(msg->parts);

    llmMessageInit(msg);
}

static int llmMessageAddPart(LlmmapiMessage* msg, const LlmmapiPart* part)
{
    LlmmapiPart* p = (LlmmapiPart*)realloc(msg->parts,
                                           (msg->partCount + 1) * sizeof(LlmmapiPart));
    if (p == NULL)
    {
        return -1;
    }

    msg->parts = p;
    msg->parts[msg->partCount++] = *part;

    return 0;
}

// Allocate a zeroed part at the end of the message; returns NULL on OOM.
static LlmmapiPart* llmMessageNewPart(LlmmapiMessage* msg)
{
    LlmmapiPart* p = (LlmmapiPart*)realloc(msg->parts,
                                           (msg->partCount + 1) * sizeof(LlmmapiPart));
    if (p == NULL)
    {
        return NULL;
    }

    msg->parts = p;
    LlmmapiPart* part = &msg->parts[msg->partCount++];
    memset(part, 0, sizeof(*part));

    return part;
}

static void llmAddTextPart(LlmmapiMessage* msg, LlmmapiPartType type,
                           const char* text)
{
    if (text == NULL)
    {
        return;
    }

    LlmmapiPart* p = llmMessageNewPart(msg);
    if (p == NULL)
    {
        return;
    }

    p->type = type;
    p->text = llmStrDup(text);
}

static int llmMessageHasPart(const LlmmapiMessage* msg, LlmmapiPartType type)
{
    for (size_t i = 0; i < msg->partCount; i++)
    {
        if (msg->parts[i].type == type)
        {
            return 1;
        }
    }

    return 0;
}

// True when the message has at least one part and every part is a RAW_BLOCK
// that does not belong to `provider` (so the whole message can be skipped).
static int llmMessageAllRawExcluded(const LlmmapiMessage* msg,
                                    const char* provider)
{
    if (msg->partCount == 0)
    {
        return 0;
    }

    for (size_t i = 0; i < msg->partCount; i++)
    {
        const LlmmapiPart* p = &msg->parts[i];
        if (p->type != LLMMAPI_PART_RAW_BLOCK)
        {
            return 0;
        }
        if (p->provider != NULL && strcmp(p->provider, provider) == 0)
        {
            return 0;
        }
    }

    return 1;
}

static char* llmConcatParts(const LlmmapiMessage* msg, LlmmapiPartType type)
{
    LlmBuf buf;
    llmBufInit(&buf);

    for (size_t i = 0; i < msg->partCount; i++)
    {
        if (msg->parts[i].type == type && msg->parts[i].text != NULL)
        {
            llmBufAppend(&buf, msg->parts[i].text);
        }
    }

    if (buf.len == 0)
    {
        llmBufFree(&buf);
        return NULL;
    }

    return llmBufDetach(&buf);
}

// ==================== Internal: Citation helpers ====================

static LlmmapiCitation* llmRespAddCitation(LlmmapiResp* resp)
{
    if (resp->citationCount >= resp->citationCap)
    {
        size_t newCap = resp->citationCap == 0 ? 4 : resp->citationCap * 2;
        LlmmapiCitation* p = (LlmmapiCitation*)realloc(
            resp->citations, newCap * sizeof(LlmmapiCitation));
        if (p == NULL)
        {
            return NULL;
        }

        resp->citations   = p;
        resp->citationCap = newCap;
    }

    LlmmapiCitation* c = &resp->citations[resp->citationCount++];
    memset(c, 0, sizeof(*c));
    c->startIndex = -1;
    c->endIndex   = -1;

    return c;
}

// ==================== Public: Message accessors ====================

size_t llmapiMessageGetPartCount(const LlmmapiMessage* msg)
{
    return msg ? msg->partCount : 0;
}

const LlmmapiPart* llmapiMessageGetPart(const LlmmapiMessage* msg, size_t idx)
{
    if (msg == NULL || idx >= msg->partCount)
    {
        return NULL;
    }

    return &msg->parts[idx];
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
    ctx->api = LLMMAPI_API_OPENAI;

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

// ==================== Transport / provider ====================

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

void llmapiSetApi(LlmmapiCtx* ctx, LlmmapiApi api)
{
    if (ctx == NULL)
    {
        return;
    }

    if (api == LLMMAPI_API_OPENAI || api == LLMMAPI_API_ANTHROPIC ||
        api == LLMMAPI_API_OPENAI_RESPONSES)
    {
        ctx->api = api;
    }
}

// ==================== Request ====================

LlmmapiReq* llmapiReqNew(const char* model)
{
    if (model == NULL)
    {
        return NULL;
    }

    LlmmapiReq* req = (LlmmapiReq*)calloc(1, sizeof(LlmmapiReq));
    if (req == NULL)
    {
        return NULL;
    }

    req->model = llmStrDup(model);
    if (req->model == NULL)
    {
        free(req);
        return NULL;
    }

    return req;
}

void llmapiReqFree(LlmmapiReq* req)
{
    if (req == NULL)
    {
        return;
    }

    free(req->model);

    for (size_t i = 0; i < req->messageCount; i++)
    {
        llmMessageFree(req->messages[i]);
        free(req->messages[i]);
    }
    free(req->messages);

    for (size_t i = 0; i < req->toolCount; i++)
    {
        free(req->tools[i].name);
        free(req->tools[i].description);
        free(req->tools[i].parametersJson);
    }
    free(req->tools);

    for (size_t i = 0; i < req->rawToolCount; i++)
    {
        free(req->rawTools[i]);
    }
    free(req->rawTools);

    free(req->stopJson);

    for (size_t i = 0; i < req->extraCount; i++)
    {
        free(req->extraKeys[i]);
        free(req->extraValues[i]);
    }
    free(req->extraKeys);
    free(req->extraValues);

    free(req);
}

LlmmapiMessage* llmapiReqAddMessage(LlmmapiReq* req, const char* role)
{
    if (req == NULL || role == NULL || llmHasCrlf(role))
    {
        return NULL;
    }

    if (req->messageCount >= req->messageCap)
    {
        size_t newCap = req->messageCap == 0 ? 4 : req->messageCap * 2;
        LlmmapiMessage** p = (LlmmapiMessage**)realloc(
            req->messages, newCap * sizeof(LlmmapiMessage*));
        if (p == NULL)
        {
            return NULL;
        }

        req->messages   = p;
        req->messageCap = newCap;
    }

    LlmmapiMessage* msg = (LlmmapiMessage*)calloc(1, sizeof(LlmmapiMessage));
    if (msg == NULL)
    {
        return NULL;
    }

    msg->role = llmStrDup(role);
    if (msg->role == NULL)
    {
        free(msg);
        return NULL;
    }

    req->messages[req->messageCount++] = msg;

    return msg;
}

void llmapiReqAddMsg(LlmmapiReq* req, const char* role, const char* text)
{
    LlmmapiMessage* msg = llmapiReqAddMessage(req, role);
    if (msg != NULL)
    {
        llmapiMsgAddText(msg, text);
    }
}

void llmapiMsgAddText(LlmmapiMessage* msg, const char* text)
{
    if (msg == NULL || text == NULL)
    {
        return;
    }

    LlmmapiPart p;
    memset(&p, 0, sizeof(p));
    p.type = LLMMAPI_PART_TEXT;
    p.text = llmStrDup(text);

    if (p.text == NULL || llmMessageAddPart(msg, &p) != 0)
    {
        llmPartFree(&p);
    }
}

void llmapiMsgAddImage(LlmmapiMessage* msg, const char* url)
{
    if (msg == NULL || url == NULL)
    {
        return;
    }

    LlmmapiPart p;
    memset(&p, 0, sizeof(p));
    p.type = LLMMAPI_PART_IMAGE;
    p.url  = llmStrDup(url);

    if (p.url == NULL || llmMessageAddPart(msg, &p) != 0)
    {
        llmPartFree(&p);
    }
}

void llmapiMsgAddFile(LlmmapiMessage* msg, const char* filename,
                      const char* mimeType, const char* dataUrlOrFileId)
{
    if (msg == NULL || dataUrlOrFileId == NULL)
    {
        return;
    }

    LlmmapiPart p;
    memset(&p, 0, sizeof(p));
    p.type     = LLMMAPI_PART_FILE;
    p.filename = llmStrDup(filename);
    p.mimeType = llmStrDup(mimeType);

    if (strncmp(dataUrlOrFileId, "data:", 5) == 0)
    {
        p.url = llmStrDup(dataUrlOrFileId);
    }
    else
    {
        p.fileId = llmStrDup(dataUrlOrFileId);
    }

    if (llmMessageAddPart(msg, &p) != 0)
    {
        llmPartFree(&p);
    }
}

void llmapiMsgAddToolCall(LlmmapiMessage* msg, const char* toolCallId,
                          const char* toolName, const char* argsJson)
{
    if (msg == NULL || toolName == NULL)
    {
        return;
    }

    LlmmapiPart p;
    memset(&p, 0, sizeof(p));
    p.type       = LLMMAPI_PART_TOOL_CALL;
    p.toolCallId = llmStrDup(toolCallId);
    p.toolName   = llmStrDup(toolName);
    p.text       = llmStrDup(argsJson ? argsJson : "{}");

    if (llmMessageAddPart(msg, &p) != 0)
    {
        llmPartFree(&p);
    }
}

void llmapiMsgAddToolResult(LlmmapiMessage* msg, const char* toolCallId,
                            const char* content)
{
    if (msg == NULL || toolCallId == NULL)
    {
        return;
    }

    LlmmapiPart p;
    memset(&p, 0, sizeof(p));
    p.type       = LLMMAPI_PART_TOOL_RESULT;
    p.toolCallId = llmStrDup(toolCallId);
    p.text       = llmStrDup(content ? content : "");

    if (llmMessageAddPart(msg, &p) != 0)
    {
        llmPartFree(&p);
    }
}

int llmapiMsgAddRawBlock(LlmmapiMessage* msg,
                         const char* provider,
                         const char* jsonObject)
{
    if (msg == NULL || provider == NULL || jsonObject == NULL)
    {
        return -1;
    }

    cJSON* parsed = cJSON_Parse(jsonObject);
    if (parsed == NULL || !cJSON_IsObject(parsed))
    {
        cJSON_Delete(parsed);
        return -1;
    }
    cJSON_Delete(parsed);

    LlmmapiPart p;
    memset(&p, 0, sizeof(p));
    p.type     = LLMMAPI_PART_RAW_BLOCK;
    p.provider = llmStrDup(provider);
    p.rawJson  = llmStrDup(jsonObject);

    if (p.provider == NULL || p.rawJson == NULL)
    {
        llmPartFree(&p);
        return -1;
    }

    if (llmMessageAddPart(msg, &p) != 0)
    {
        llmPartFree(&p);
        return -1;
    }

    return 0;
}

void llmapiReqAddTool(LlmmapiReq* req, const char* name,
                      const char* description, const char* parametersJson)
{
    if (req == NULL || name == NULL)
    {
        return;
    }

    if (req->toolCount >= req->toolCap)
    {
        size_t newCap = req->toolCap == 0 ? 4 : req->toolCap * 2;
        LlmTool* p = (LlmTool*)realloc(req->tools, newCap * sizeof(LlmTool));
        if (p == NULL)
        {
            return;
        }

        req->tools   = p;
        req->toolCap = newCap;
    }

    LlmTool tool;
    tool.name           = llmStrDup(name);
    tool.description    = llmStrDup(description);
    tool.parametersJson = llmStrDup(parametersJson);

    if (tool.name == NULL)
    {
        free(tool.description);
        free(tool.parametersJson);
        return;
    }

    req->tools[req->toolCount++] = tool;
}

int llmapiReqAddRawTool(LlmmapiReq* req, const char* jsonObject)
{
    if (req == NULL || jsonObject == NULL)
    {
        return -1;
    }

    cJSON* parsed = cJSON_Parse(jsonObject);
    if (parsed == NULL || !cJSON_IsObject(parsed))
    {
        cJSON_Delete(parsed);
        return -1;
    }
    cJSON_Delete(parsed);

    if (req->rawToolCount >= req->rawToolCap)
    {
        size_t newCap = req->rawToolCap == 0 ? 4 : req->rawToolCap * 2;
        char** p = (char**)realloc(req->rawTools, newCap * sizeof(char*));
        if (p == NULL)
        {
            return -1;
        }

        req->rawTools   = p;
        req->rawToolCap = newCap;
    }

    char* copy = llmStrDup(jsonObject);
    if (copy == NULL)
    {
        return -1;
    }

    req->rawTools[req->rawToolCount++] = copy;

    return 0;
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

void llmapiReqSetTopP(LlmmapiReq* req, double topP)
{
    if (req == NULL)
    {
        return;
    }

    req->hasTopP = 1;
    req->topP    = topP;
}

void llmapiReqSetStop(LlmmapiReq* req, const char* jsonArray)
{
    if (req == NULL)
    {
        return;
    }

    free(req->stopJson);
    req->stopJson = llmStrDup(jsonArray);
}

void llmapiReqSetExtra(LlmmapiReq* req, const char* key, const char* jsonValue)
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

    req->extraKeys[req->extraCount]   = llmStrDup(key);
    req->extraValues[req->extraCount] = llmStrDup(jsonValue);

    if (req->extraKeys[req->extraCount] == NULL ||
        req->extraValues[req->extraCount] == NULL)
    {
        free(req->extraKeys[req->extraCount]);
        free(req->extraValues[req->extraCount]);
        return;
    }

    req->extraCount++;
}

// ==================== Internal: JSON helpers ====================

static const char* llmJsonStr(const cJSON* obj, const char* key)
{
    cJSON* item = cJSON_GetObjectItem(obj, key);
    if (item != NULL && cJSON_IsString(item))
    {
        return item->valuestring;
    }

    return NULL;
}

static int llmJsonLong(const cJSON* obj, const char* key, long* out)
{
    cJSON* item = cJSON_GetObjectItem(obj, key);
    if (item != NULL && cJSON_IsNumber(item))
    {
        *out = (long)item->valuedouble;
        return 1;
    }

    return 0;
}

// Parse "data:<mime>;base64,<payload>".  On success, *mimeOut is malloc'd and
// *dataOut points into `url` (not allocated).
static int llmParseDataUrl(const char* url, char** mimeOut, const char** dataOut)
{
    if (url == NULL || strncmp(url, "data:", 5) != 0)
    {
        return 0;
    }

    const char* p    = url + 5;
    const char* semi = strchr(p, ';');
    if (semi == NULL || strncmp(semi, ";base64,", 8) != 0)
    {
        return 0;
    }

    char* mime = llmStrNDup(p, (size_t)(semi - p));
    if (mime == NULL)
    {
        return 0;
    }

    *mimeOut = mime;
    *dataOut = semi + 8;

    return 1;
}

// Append a parsed JSON value, falling back to a string when parsing fails.
static void llmAddJsonValue(cJSON* root, const char* key, const char* value)
{
    cJSON* parsed = cJSON_Parse(value);
    if (parsed == NULL)
    {
        parsed = cJSON_CreateString(value);
    }

    if (parsed != NULL)
    {
        cJSON_AddItemToObject(root, key, parsed);
    }
}

// Append provider-native tool objects verbatim to a "tools" array.
static void llmAppendRawTools(const LlmmapiReq* req, cJSON* tools)
{
    for (size_t i = 0; i < req->rawToolCount; i++)
    {
        cJSON* t = cJSON_Parse(req->rawTools[i]);
        if (t != NULL && cJSON_IsObject(t))
        {
            cJSON_AddItemToArray(tools, t);
        }
        else
        {
            cJSON_Delete(t);
        }
    }
}

// ==================== Internal: Provider interface ====================

#define LLMAPI_MAX_SERVER_TOOLS 64

typedef struct
{
    LlmmapiStreamCb cb;
    void*           userdata;
    int             done;
    long            inputTokens;
    long            outputTokens;

    int             hadCitation;
    size_t serverToolCount;
    size_t serverToolIndex[LLMAPI_MAX_SERVER_TOOLS];
    char*  serverToolName[LLMAPI_MAX_SERVER_TOOLS];
} LlmStreamSink;

static void llmEmit(LlmStreamSink* sink, LlmmapiStreamType type,
                    const char* text, const char* toolCallId,
                    const char* toolName, size_t toolIndex)
{
    LlmmapiStreamEvent ev;
    ev.type       = type;
    ev.text       = text;
    ev.toolCallId = toolCallId;
    ev.toolName   = toolName;
    ev.toolIndex  = toolIndex;
    ev.citation   = NULL;
    ev.status     = NULL;

    sink->cb(&ev, sink->userdata);
}

static void llmEmitCitation(LlmStreamSink* sink,
                            const LlmmapiCitation* citation)
{
    LlmmapiStreamEvent ev;
    memset(&ev, 0, sizeof(ev));
    ev.type     = LLMAPI_STREAM_CITATION;
    ev.citation = citation;

    sink->hadCitation = 1;
    sink->cb(&ev, sink->userdata);
}

static void llmEmitServerTool(LlmStreamSink* sink, const char* name,
                              const char* status, size_t index)
{
    LlmmapiStreamEvent ev;
    memset(&ev, 0, sizeof(ev));
    ev.type      = LLMAPI_STREAM_SERVER_TOOL;
    ev.text      = name;
    ev.status    = status;
    ev.toolIndex = index;

    sink->cb(&ev, sink->userdata);
}

static void llmEmitCitationArray(cJSON* annotations, LlmStreamSink* sink)
{
    if (annotations == NULL || !cJSON_IsArray(annotations))
    {
        return;
    }

    cJSON* a = NULL;
    cJSON_ArrayForEach(a, annotations)
    {
        const char* atype = llmJsonStr(a, "type");
        if (atype == NULL || strcmp(atype, "url_citation") != 0)
        {
            continue;
        }

        const char* url = llmJsonStr(a, "url");
        if (url == NULL)
        {
            continue;
        }

        LlmmapiCitation c;
        memset(&c, 0, sizeof(c));
        c.url        = url;
        c.title      = llmJsonStr(a, "title");
        c.startIndex = -1;
        c.endIndex   = -1;
        llmJsonLong(a, "start_index", &c.startIndex);
        llmJsonLong(a, "end_index", &c.endIndex);

        llmEmitCitation(sink, &c);
    }
}

static void llmSinkRememberServerTool(LlmStreamSink* sink, size_t index,
                                      const char* name)
{
    if (sink->serverToolCount >= LLMAPI_MAX_SERVER_TOOLS)
    {
        return;
    }

    sink->serverToolIndex[sink->serverToolCount] = index;
    sink->serverToolName[sink->serverToolCount]  = llmStrDup(name);
    sink->serverToolCount++;
}

static const char* llmSinkFindServerTool(const LlmStreamSink* sink,
                                         size_t index)
{
    for (size_t i = 0; i < sink->serverToolCount; i++)
    {
        if (sink->serverToolIndex[i] == index)
        {
            return sink->serverToolName[i];
        }
    }

    return NULL;
}

static void llmSinkFreeServerTools(LlmStreamSink* sink)
{
    for (size_t i = 0; i < sink->serverToolCount; i++)
    {
        free(sink->serverToolName[i]);
        sink->serverToolName[i] = NULL;
    }
    sink->serverToolCount = 0;
}

typedef struct
{
    const char*        path;
    const char*        authHeader;
    const char*        authPrefix;
    const char* const* staticHeaders;

    int  (*encode)(LlmmapiCtx* ctx, const LlmmapiReq* req, int stream,
                   LlmBuf* body);
    int  (*decode)(LlmmapiCtx* ctx, const char* rawJson, LlmmapiResp* out);
    void (*streamDecode)(const char* payload, LlmStreamSink* sink);
} LlmProvider;

// ==================== Provider: OpenAI ====================

static void openaiAddContentPart(cJSON* content, const LlmmapiPart* part)
{
    if (part->type == LLMMAPI_PART_TEXT)
    {
        cJSON* b = cJSON_CreateObject();
        if (b == NULL) return;
        cJSON_AddStringToObject(b, "type", "text");
        cJSON_AddStringToObject(b, "text", part->text ? part->text : "");
        cJSON_AddItemToArray(content, b);
    }
    else if (part->type == LLMMAPI_PART_IMAGE)
    {
        cJSON* b = cJSON_CreateObject();
        if (b == NULL) return;
        cJSON_AddStringToObject(b, "type", "image_url");
        cJSON* iu = cJSON_AddObjectToObject(b, "image_url");
        if (iu != NULL)
        {
            cJSON_AddStringToObject(iu, "url", part->url ? part->url : "");
        }
        cJSON_AddItemToArray(content, b);
    }
    else if (part->type == LLMMAPI_PART_FILE)
    {
        cJSON* b = cJSON_CreateObject();
        if (b == NULL) return;
        cJSON_AddStringToObject(b, "type", "file");
        cJSON* f = cJSON_AddObjectToObject(b, "file");
        if (f != NULL)
        {
            if (part->fileId != NULL)
            {
                cJSON_AddStringToObject(f, "file_id", part->fileId);
            }
            else
            {
                if (part->filename != NULL)
                {
                    cJSON_AddStringToObject(f, "filename", part->filename);
                }
                if (part->url != NULL)
                {
                    cJSON_AddStringToObject(f, "file_data", part->url);
                }
            }
        }
        cJSON_AddItemToArray(content, b);
    }
    else if (part->type == LLMMAPI_PART_RAW_BLOCK)
    {
        if (part->provider != NULL && part->rawJson != NULL &&
            strcmp(part->provider, "openai") == 0)
        {
            cJSON* obj = cJSON_Parse(part->rawJson);
            if (obj != NULL && cJSON_IsObject(obj))
            {
                cJSON_AddItemToArray(content, obj);
            }
            else
            {
                cJSON_Delete(obj);
            }
        }
    }
}

static int openaiEncodeMessage(const LlmmapiMessage* msg, cJSON* msgs)
{
    if (llmMessageAllRawExcluded(msg, "openai"))
    {
        return 0;
    }

    // Tool results become one "tool" message per result.
    if (llmMessageHasPart(msg, LLMMAPI_PART_TOOL_RESULT) ||
        strcmp(msg->role, LLMAPI_ROLE_TOOL) == 0)
    {
        for (size_t i = 0; i < msg->partCount; i++)
        {
            const LlmmapiPart* part = &msg->parts[i];
            if (part->type != LLMMAPI_PART_TOOL_RESULT)
            {
                continue;
            }

            cJSON* m = cJSON_CreateObject();
            if (m == NULL) return -1;
            cJSON_AddStringToObject(m, "role", "tool");
            cJSON_AddStringToObject(m, "tool_call_id",
                                    part->toolCallId ? part->toolCallId : "");
            cJSON_AddStringToObject(m, "content", part->text ? part->text : "");
            cJSON_AddItemToArray(msgs, m);
        }
        return 0;
    }

    // Assistant tool calls.
    if (llmMessageHasPart(msg, LLMMAPI_PART_TOOL_CALL))
    {
        cJSON* m = cJSON_CreateObject();
        if (m == NULL) return -1;
        cJSON_AddStringToObject(m, "role", "assistant");

        char* text = llmConcatParts(msg, LLMMAPI_PART_TEXT);
        if (text != NULL)
        {
            cJSON_AddStringToObject(m, "content", text);
            free(text);
        }
        else
        {
            cJSON_AddNullToObject(m, "content");
        }

        cJSON* tcs = cJSON_AddArrayToObject(m, "tool_calls");
        if (tcs != NULL)
        {
            for (size_t i = 0; i < msg->partCount; i++)
            {
                const LlmmapiPart* part = &msg->parts[i];
                if (part->type != LLMMAPI_PART_TOOL_CALL)
                {
                    continue;
                }

                cJSON* tc = cJSON_CreateObject();
                if (tc == NULL) continue;
                cJSON_AddStringToObject(tc, "id",
                                        part->toolCallId ? part->toolCallId : "");
                cJSON_AddStringToObject(tc, "type", "function");
                cJSON* fn = cJSON_AddObjectToObject(tc, "function");
                if (fn != NULL)
                {
                    cJSON_AddStringToObject(fn, "name",
                                            part->toolName ? part->toolName : "");
                    cJSON_AddStringToObject(fn, "arguments",
                                            part->text ? part->text : "{}");
                }
                cJSON_AddItemToArray(tcs, tc);
            }
        }

        cJSON_AddItemToArray(msgs, m);
        return 0;
    }

    // Ordinary message.
    cJSON* m = cJSON_CreateObject();
    if (m == NULL) return -1;
    cJSON_AddStringToObject(m, "role", msg->role);

    int hasMultipart = 0;
    for (size_t i = 0; i < msg->partCount; i++)
    {
        LlmmapiPartType t = msg->parts[i].type;
        if (t == LLMMAPI_PART_IMAGE || t == LLMMAPI_PART_FILE ||
            t == LLMMAPI_PART_RAW_BLOCK)
        {
            hasMultipart = 1;
            break;
        }
    }

    if (!hasMultipart)
    {
        char* text = llmConcatParts(msg, LLMMAPI_PART_TEXT);
        cJSON_AddStringToObject(m, "content", text ? text : "");
        free(text);
    }
    else
    {
        cJSON* content = cJSON_AddArrayToObject(m, "content");
        if (content != NULL)
        {
            for (size_t i = 0; i < msg->partCount; i++)
            {
                openaiAddContentPart(content, &msg->parts[i]);
            }
        }
    }

    cJSON_AddItemToArray(msgs, m);

    return 0;
}

static void openaiAddTools(const LlmmapiReq* req, cJSON* root)
{
    if (req->toolCount == 0 && req->rawToolCount == 0)
    {
        return;
    }

    cJSON* tools = cJSON_AddArrayToObject(root, "tools");
    if (tools == NULL)
    {
        return;
    }

    for (size_t i = 0; i < req->toolCount; i++)
    {
        const LlmTool* tool = &req->tools[i];

        cJSON* t = cJSON_CreateObject();
        if (t == NULL) continue;
        cJSON_AddStringToObject(t, "type", "function");
        cJSON* fn = cJSON_AddObjectToObject(t, "function");
        if (fn != NULL)
        {
            cJSON_AddStringToObject(fn, "name", tool->name ? tool->name : "");
            if (tool->description != NULL)
            {
                cJSON_AddStringToObject(fn, "description", tool->description);
            }

            cJSON* schema = tool->parametersJson
                                ? cJSON_Parse(tool->parametersJson)
                                : NULL;
            if (schema == NULL)
            {
                schema = cJSON_CreateObject();
            }
            if (schema != NULL)
            {
                cJSON_AddItemToObject(fn, "parameters", schema);
            }
        }
        cJSON_AddItemToArray(tools, t);
    }

    llmAppendRawTools(req, tools);
}

static void llmAddCommonParams(const LlmmapiReq* req, cJSON* root, int stream,
                               const char* stopKey)
{
    cJSON_AddBoolToObject(root, "stream", stream ? 1 : 0);

    if (req->hasMaxTokens)
    {
        cJSON_AddNumberToObject(root, "max_tokens", (double)req->maxTokens);
    }

    if (req->hasTemperature)
    {
        cJSON_AddNumberToObject(root, "temperature", req->temperature);
    }

    if (req->hasTopP)
    {
        cJSON_AddNumberToObject(root, "top_p", req->topP);
    }

    if (req->stopJson != NULL)
    {
        cJSON* stop = cJSON_Parse(req->stopJson);
        if (stop != NULL)
        {
            cJSON_AddItemToObject(root, stopKey, stop);
        }
    }

    for (size_t i = 0; i < req->extraCount; i++)
    {
        llmAddJsonValue(root, req->extraKeys[i], req->extraValues[i]);
    }
}

static int openaiEncode(LlmmapiCtx* ctx, const LlmmapiReq* req, int stream,
                        LlmBuf* body)
{
    (void)ctx;

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

    for (size_t i = 0; i < req->messageCount; i++)
    {
        if (openaiEncodeMessage(req->messages[i], msgs) != 0)
        {
            cJSON_Delete(root);
            return -1;
        }
    }

    openaiAddTools(req, root);

    llmAddCommonParams(req, root, stream, "stop");

    char* json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (json == NULL)
    {
        return -1;
    }

    int rc = llmBufAppend(body, json);
    free(json);

    return rc;
}

static int openaiDecode(LlmmapiCtx* ctx, const char* rawJson, LlmmapiResp* out)
{
    (void)ctx;
    cJSON* root = cJSON_Parse(rawJson);
    if (root == NULL)
    {
        return -1;
    }

    out->rawJson = llmStrDup(rawJson);

    cJSON* choices = cJSON_GetObjectItem(root, "choices");
    if (choices != NULL && cJSON_IsArray(choices) &&
        cJSON_GetArraySize(choices) > 0)
    {
        cJSON* choice  = cJSON_GetArrayItem(choices, 0);
        cJSON* message = cJSON_GetObjectItem(choice, "message");

        if (message != NULL)
        {
            cJSON* content = cJSON_GetObjectItem(message, "content");
            if (content != NULL && cJSON_IsString(content))
            {
                llmAddTextPart(&out->message, LLMMAPI_PART_TEXT,
                               content->valuestring);
            }
            else if (content != NULL && cJSON_IsArray(content))
            {
                cJSON* part = NULL;
                cJSON_ArrayForEach(part, content)
                {
                    const char* type = llmJsonStr(part, "type");
                    if (type != NULL && strcmp(type, "text") == 0)
                    {
                        llmAddTextPart(&out->message, LLMMAPI_PART_TEXT,
                                       llmJsonStr(part, "text"));
                    }
                }
            }

            const char* reasoning = llmJsonStr(message, "reasoning_content");
            if (reasoning != NULL)
            {
                llmAddTextPart(&out->message, LLMMAPI_PART_THINKING, reasoning);
            }

            cJSON* tcs = cJSON_GetObjectItem(message, "tool_calls");
            if (tcs != NULL && cJSON_IsArray(tcs))
            {
                cJSON* tc = NULL;
                cJSON_ArrayForEach(tc, tcs)
                {
                    cJSON* fn = cJSON_GetObjectItem(tc, "function");

                    LlmmapiPart* p = llmMessageNewPart(&out->message);
                    if (p == NULL) continue;

                    p->type       = LLMMAPI_PART_TOOL_CALL;
                    p->toolCallId = llmStrDup(llmJsonStr(tc, "id"));
                    p->toolName   = llmStrDup(fn ? llmJsonStr(fn, "name") : NULL);
                    p->text       = llmStrDup(fn ? llmJsonStr(fn, "arguments")
                                                 : NULL);
                    if (p->text == NULL)
                    {
                        p->text = llmStrDup("{}");
                    }
                }
            }
        }

        const char* finish = llmJsonStr(choice, "finish_reason");
        if (finish != NULL)
        {
            out->finishReason = llmStrDup(finish);
        }
    }

    cJSON* usage = cJSON_GetObjectItem(root, "usage");
    long in = 0, outTok = 0, total = 0;
    if (usage != NULL)
    {
        llmJsonLong(usage, "prompt_tokens", &in);
        llmJsonLong(usage, "completion_tokens", &outTok);
        if (!llmJsonLong(usage, "total_tokens", &total))
        {
            total = in + outTok;
        }
    }

    out->inputTokens  = in;
    out->outputTokens = outTok;
    out->totalTokens  = total;

    cJSON_Delete(root);

    return 0;
}

static void openaiStreamDecode(const char* payload, LlmStreamSink* sink)
{
    if (strcmp(payload, "[DONE]") == 0)
    {
        sink->done = 1;
        llmEmit(sink, LLMAPI_STREAM_DONE, NULL, NULL, NULL, 0);
        return;
    }

    cJSON* root = cJSON_Parse(payload);
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
            const char* content = llmJsonStr(delta, "content");
            if (content != NULL && content[0] != '\0')
            {
                llmEmit(sink, LLMAPI_STREAM_TEXT, content, NULL, NULL, 0);
            }

            const char* reasoning = llmJsonStr(delta, "reasoning_content");
            if (reasoning != NULL && reasoning[0] != '\0')
            {
                llmEmit(sink, LLMAPI_STREAM_THINKING, reasoning, NULL, NULL, 0);
            }

            cJSON* tcs = cJSON_GetObjectItem(delta, "tool_calls");
            if (tcs != NULL && cJSON_IsArray(tcs))
            {
                size_t fallback = 0;
                cJSON* tc = NULL;
                cJSON_ArrayForEach(tc, tcs)
                {
                    size_t index = fallback;
                    cJSON* idx = cJSON_GetObjectItem(tc, "index");
                    if (idx != NULL && cJSON_IsNumber(idx))
                    {
                        index = (size_t)idx->valuedouble;
                    }
                    fallback = index + 1;

                    cJSON* fn = cJSON_GetObjectItem(tc, "function");
                    const char* id   = llmJsonStr(tc, "id");
                    const char* name = fn ? llmJsonStr(fn, "name") : NULL;
                    const char* args = fn ? llmJsonStr(fn, "arguments") : NULL;

                    llmEmit(sink, LLMAPI_STREAM_TOOL_CALL, args, id, name,
                            index);
                }
            }
        }
    }

    cJSON* usage = cJSON_GetObjectItem(root, "usage");
    if (usage != NULL)
    {
        long in = 0, outTok = 0;
        if (llmJsonLong(usage, "prompt_tokens", &in))
        {
            sink->inputTokens = in;
        }
        if (llmJsonLong(usage, "completion_tokens", &outTok))
        {
            sink->outputTokens = outTok;
        }
    }

    cJSON_Delete(root);
}

// ==================== Provider: Anthropic ====================

static int anthropicAddImage(cJSON* content, const LlmmapiPart* part)
{
    if (part->url == NULL)
    {
        return -1;
    }

    cJSON* block = cJSON_CreateObject();
    if (block == NULL)
    {
        return -1;
    }
    cJSON_AddStringToObject(block, "type", "image");

    cJSON* source = cJSON_AddObjectToObject(block, "source");
    if (source == NULL)
    {
        cJSON_Delete(block);
        return -1;
    }

    char* mime = NULL;
    const char* data = NULL;
    if (llmParseDataUrl(part->url, &mime, &data))
    {
        cJSON_AddStringToObject(source, "type", "base64");
        cJSON_AddStringToObject(source, "media_type", mime);
        cJSON_AddStringToObject(source, "data", data);
        free(mime);
    }
    else
    {
        cJSON_AddStringToObject(source, "type", "url");
        cJSON_AddStringToObject(source, "url", part->url);
    }

    cJSON_AddItemToArray(content, block);

    return 0;
}

static int anthropicAddFile(cJSON* content, const LlmmapiPart* part)
{
    // Anthropic has no provider-side file id source; require an inline data URL.
    if (part->url == NULL)
    {
        return -1;
    }

    char* mime = NULL;
    const char* data = NULL;
    if (!llmParseDataUrl(part->url, &mime, &data))
    {
        return -1;
    }

    cJSON* block = cJSON_CreateObject();
    if (block == NULL)
    {
        free(mime);
        return -1;
    }
    cJSON_AddStringToObject(block, "type", "document");

    cJSON* source = cJSON_AddObjectToObject(block, "source");
    if (source == NULL)
    {
        free(mime);
        cJSON_Delete(block);
        return -1;
    }

    cJSON_AddStringToObject(source, "type", "base64");
    cJSON_AddStringToObject(source, "media_type", mime);
    cJSON_AddStringToObject(source, "data", data);

    free(mime);

    cJSON_AddItemToArray(content, block);

    return 0;
}

static int anthropicEncodeMessage(LlmmapiCtx* ctx, const LlmmapiMessage* msg,
                                  cJSON* msgs)
{
    if (llmMessageAllRawExcluded(msg, "anthropic"))
    {
        return 0;
    }

    int isTool = (strcmp(msg->role, LLMAPI_ROLE_TOOL) == 0) ||
                 llmMessageHasPart(msg, LLMMAPI_PART_TOOL_RESULT);

    if (isTool)
    {
        // Handled by the caller (which merges consecutive tool messages).
        (void)ctx;
        return 0;
    }

    cJSON* m = cJSON_CreateObject();
    if (m == NULL)
    {
        return -1;
    }

    const char* role = (strcmp(msg->role, LLMAPI_ROLE_ASSISTANT) == 0)
                           ? "assistant"
                           : "user";
    cJSON_AddStringToObject(m, "role", role);

    cJSON* content = cJSON_AddArrayToObject(m, "content");
    if (content == NULL)
    {
        cJSON_Delete(m);
        return -1;
    }

    for (size_t i = 0; i < msg->partCount; i++)
    {
        const LlmmapiPart* part = &msg->parts[i];

        if (part->type == LLMMAPI_PART_TEXT)
        {
            cJSON* b = cJSON_CreateObject();
            if (b == NULL) continue;
            cJSON_AddStringToObject(b, "type", "text");
            cJSON_AddStringToObject(b, "text", part->text ? part->text : "");
            cJSON_AddItemToArray(content, b);
        }
        else if (part->type == LLMMAPI_PART_IMAGE)
        {
            if (anthropicAddImage(content, part) != 0)
            {
                llmBufAppend(&ctx->errbuf,
                             "Anthropic: image part needs a url");
                cJSON_Delete(m);
                return -1;
            }
        }
        else if (part->type == LLMMAPI_PART_FILE)
        {
            if (anthropicAddFile(content, part) != 0)
            {
                llmBufAppend(&ctx->errbuf,
                             "Anthropic: file part needs an inline data URL");
                cJSON_Delete(m);
                return -1;
            }
        }
        else if (part->type == LLMMAPI_PART_TOOL_CALL)
        {
            cJSON* b = cJSON_CreateObject();
            if (b == NULL) continue;
            cJSON_AddStringToObject(b, "type", "tool_use");
            cJSON_AddStringToObject(b, "id",
                                    part->toolCallId ? part->toolCallId : "");
            cJSON_AddStringToObject(b, "name",
                                    part->toolName ? part->toolName : "");

            cJSON* input = part->text ? cJSON_Parse(part->text) : NULL;
            if (input == NULL)
            {
                input = cJSON_CreateObject();
            }
            if (input != NULL)
            {
                cJSON_AddItemToObject(b, "input", input);
            }

            cJSON_AddItemToArray(content, b);
        }
        else if (part->type == LLMMAPI_PART_RAW_BLOCK)
        {
            if (part->provider != NULL && part->rawJson != NULL &&
                strcmp(part->provider, "anthropic") == 0)
            {
                cJSON* obj = cJSON_Parse(part->rawJson);
                if (obj != NULL && cJSON_IsObject(obj))
                {
                    cJSON_AddItemToArray(content, obj);
                }
                else
                {
                    cJSON_Delete(obj);
                }
            }
        }
        // THINKING and TOOL_RESULT are not sent back as input.
    }

    cJSON_AddItemToArray(msgs, m);

    return 0;
}

static void anthropicAddTools(const LlmmapiReq* req, cJSON* root)
{
    if (req->toolCount == 0 && req->rawToolCount == 0)
    {
        return;
    }

    cJSON* tools = cJSON_AddArrayToObject(root, "tools");
    if (tools == NULL)
    {
        return;
    }

    for (size_t i = 0; i < req->toolCount; i++)
    {
        const LlmTool* tool = &req->tools[i];

        cJSON* t = cJSON_CreateObject();
        if (t == NULL) continue;
        cJSON_AddStringToObject(t, "name", tool->name ? tool->name : "");
        if (tool->description != NULL)
        {
            cJSON_AddStringToObject(t, "description", tool->description);
        }

        cJSON* schema = tool->parametersJson
                            ? cJSON_Parse(tool->parametersJson)
                            : NULL;
        if (schema == NULL)
        {
            schema = cJSON_CreateObject();
        }
        if (schema != NULL)
        {
            cJSON_AddItemToObject(t, "input_schema", schema);
        }

        cJSON_AddItemToArray(tools, t);
    }

    llmAppendRawTools(req, tools);
}

static int anthropicEncode(LlmmapiCtx* ctx, const LlmmapiReq* req, int stream,
                           LlmBuf* body)
{
    cJSON* root = cJSON_CreateObject();
    if (root == NULL)
    {
        return -1;
    }

    cJSON_AddStringToObject(root, "model", req->model);

    // System prompt is a top-level field in the Anthropic API.
    LlmBuf system;
    llmBufInit(&system);
    for (size_t i = 0; i < req->messageCount; i++)
    {
        const LlmmapiMessage* msg = req->messages[i];
        if (strcmp(msg->role, LLMAPI_ROLE_SYSTEM) != 0)
        {
            continue;
        }

        for (size_t j = 0; j < msg->partCount; j++)
        {
            if (msg->parts[j].type == LLMMAPI_PART_TEXT &&
                msg->parts[j].text != NULL)
            {
                if (system.len > 0)
                {
                    llmBufAppend(&system, "\n\n");
                }
                llmBufAppend(&system, msg->parts[j].text);
            }
        }
    }
    if (system.len > 0)
    {
        cJSON_AddStringToObject(root, "system", system.data);
    }
    llmBufFree(&system);

    cJSON* msgs = cJSON_AddArrayToObject(root, "messages");
    if (msgs == NULL)
    {
        cJSON_Delete(root);
        return -1;
    }

    for (size_t i = 0; i < req->messageCount; i++)
    {
        const LlmmapiMessage* msg = req->messages[i];

        if (strcmp(msg->role, LLMAPI_ROLE_SYSTEM) == 0)
        {
            continue;
        }

        int isTool = (strcmp(msg->role, LLMAPI_ROLE_TOOL) == 0) ||
                     llmMessageHasPart(msg, LLMMAPI_PART_TOOL_RESULT);

        if (isTool)
        {
            // Merge consecutive tool-result messages into one user message.
            cJSON* m = cJSON_CreateObject();
            if (m == NULL)
            {
                cJSON_Delete(root);
                return -1;
            }
            cJSON_AddStringToObject(m, "role", "user");
            cJSON* content = cJSON_AddArrayToObject(m, "content");
            if (content == NULL)
            {
                cJSON_Delete(m);
                cJSON_Delete(root);
                return -1;
            }

            size_t j = i;
            while (j < req->messageCount)
            {
                const LlmmapiMessage* tm = req->messages[j];
                int tj = (strcmp(tm->role, LLMAPI_ROLE_TOOL) == 0) ||
                         llmMessageHasPart(tm, LLMMAPI_PART_TOOL_RESULT);
                if (!tj)
                {
                    break;
                }

                for (size_t k = 0; k < tm->partCount; k++)
                {
                    const LlmmapiPart* part = &tm->parts[k];
                    if (part->type != LLMMAPI_PART_TOOL_RESULT)
                    {
                        continue;
                    }

                    cJSON* tr = cJSON_CreateObject();
                    if (tr == NULL) continue;
                    cJSON_AddStringToObject(tr, "type", "tool_result");
                    cJSON_AddStringToObject(tr, "tool_use_id",
                                            part->toolCallId
                                                ? part->toolCallId
                                                : "");
                    cJSON_AddStringToObject(tr, "content",
                                            part->text ? part->text : "");
                    cJSON_AddItemToArray(content, tr);
                }
                j++;
            }

            i = j - 1;
            cJSON_AddItemToArray(msgs, m);
            continue;
        }

        if (anthropicEncodeMessage(ctx, msg, msgs) != 0)
        {
            cJSON_Delete(root);
            return -1;
        }
    }

    anthropicAddTools(req, root);

    // max_tokens is required by the Anthropic API.
    if (req->hasMaxTokens)
    {
        cJSON_AddNumberToObject(root, "max_tokens", (double)req->maxTokens);
    }
    else
    {
        cJSON_AddNumberToObject(root, "max_tokens",
                                (double)LLMAPI_DEFAULT_MAX_TOKENS);
    }

    if (req->hasTemperature)
    {
        cJSON_AddNumberToObject(root, "temperature", req->temperature);
    }
    if (req->hasTopP)
    {
        cJSON_AddNumberToObject(root, "top_p", req->topP);
    }
    if (req->stopJson != NULL)
    {
        cJSON* stop = cJSON_Parse(req->stopJson);
        if (stop != NULL)
        {
            cJSON_AddItemToObject(root, "stop_sequences", stop);
        }
    }

    cJSON_AddBoolToObject(root, "stream", stream ? 1 : 0);

    for (size_t i = 0; i < req->extraCount; i++)
    {
        llmAddJsonValue(root, req->extraKeys[i], req->extraValues[i]);
    }

    char* json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (json == NULL)
    {
        return -1;
    }

    int rc = llmBufAppend(body, json);
    free(json);

    return rc;
}

static int anthropicDecode(LlmmapiCtx* ctx, const char* rawJson,
                           LlmmapiResp* out)
{
    (void)ctx;
    cJSON* root = cJSON_Parse(rawJson);
    if (root == NULL)
    {
        return -1;
    }

    out->rawJson = llmStrDup(rawJson);

    cJSON* content = cJSON_GetObjectItem(root, "content");
    if (content != NULL && cJSON_IsArray(content))
    {
        cJSON* block = NULL;
        cJSON_ArrayForEach(block, content)
        {
            const char* type = llmJsonStr(block, "type");
            if (type == NULL)
            {
                continue;
            }

            if (strcmp(type, "text") == 0)
            {
                llmAddTextPart(&out->message, LLMMAPI_PART_TEXT,
                               llmJsonStr(block, "text"));

                cJSON* citations = cJSON_GetObjectItem(block, "citations");
                if (citations != NULL && cJSON_IsArray(citations))
                {
                    cJSON* cit = NULL;
                    cJSON_ArrayForEach(cit, citations)
                    {
                        const char* url = llmJsonStr(cit, "url");
                        if (url == NULL)
                        {
                            continue;
                        }

                        LlmmapiCitation* c = llmRespAddCitation(out);
                        if (c == NULL)
                        {
                            continue;
                        }

                        c->url       = llmStrDup(url);
                        c->title     = llmStrDup(llmJsonStr(cit, "title"));
                        c->citedText = llmStrDup(llmJsonStr(cit, "cited_text"));
                        llmJsonLong(cit, "start_char_index", &c->startIndex);
                        llmJsonLong(cit, "end_char_index", &c->endIndex);
                    }
                }
            }
            else if (strcmp(type, "thinking") == 0)
            {
                llmAddTextPart(&out->message, LLMMAPI_PART_THINKING,
                               llmJsonStr(block, "thinking"));
            }
            else if (strcmp(type, "tool_use") == 0)
            {
                LlmmapiPart* p = llmMessageNewPart(&out->message);
                if (p == NULL) continue;

                p->type       = LLMMAPI_PART_TOOL_CALL;
                p->toolCallId = llmStrDup(llmJsonStr(block, "id"));
                p->toolName   = llmStrDup(llmJsonStr(block, "name"));

                cJSON* input = cJSON_GetObjectItem(block, "input");
                p->text = input ? cJSON_PrintUnformatted(input)
                                : llmStrDup("{}");
                if (p->text == NULL)
                {
                    p->text = llmStrDup("{}");
                }
            }
            else
            {
                // Preserve provider-native blocks (server_tool_use,
                // web_search_tool_result, redacted_thinking, ...) so callers
                // can replay them in later turns.
                char* raw = cJSON_PrintUnformatted(block);
                if (raw != NULL)
                {
                    LlmmapiPart* p = llmMessageNewPart(&out->message);
                    if (p != NULL)
                    {
                        p->type     = LLMMAPI_PART_RAW_BLOCK;
                        p->provider = llmStrDup("anthropic");
                        p->rawJson  = raw;
                        raw = NULL;
                    }
                    free(raw);
                }
            }
        }
    }

    const char* stop = llmJsonStr(root, "stop_reason");
    if (stop != NULL)
    {
        out->finishReason = llmStrDup(stop);
    }

    cJSON* usage = cJSON_GetObjectItem(root, "usage");
    long in = 0, outTok = 0;
    if (usage != NULL)
    {
        llmJsonLong(usage, "input_tokens", &in);
        llmJsonLong(usage, "output_tokens", &outTok);
    }

    out->inputTokens  = in;
    out->outputTokens = outTok;
    out->totalTokens  = in + outTok;

    cJSON_Delete(root);

    return 0;
}

static void anthropicStreamDecode(const char* payload, LlmStreamSink* sink)
{
    cJSON* root = cJSON_Parse(payload);
    if (root == NULL)
    {
        return;
    }

    const char* type = llmJsonStr(root, "type");
    if (type == NULL)
    {
        cJSON_Delete(root);
        return;
    }

    if (strcmp(type, "message_start") == 0)
    {
        cJSON* message = cJSON_GetObjectItem(root, "message");
        cJSON* usage   = message ? cJSON_GetObjectItem(message, "usage") : NULL;
        long in = 0;
        if (usage != NULL && llmJsonLong(usage, "input_tokens", &in))
        {
            sink->inputTokens = in;
        }
    }
    else if (strcmp(type, "content_block_start") == 0)
    {
        cJSON* block = cJSON_GetObjectItem(root, "content_block");
        size_t index = 0;
        cJSON* idx = cJSON_GetObjectItem(root, "index");
        if (idx != NULL && cJSON_IsNumber(idx))
        {
            index = (size_t)idx->valuedouble;
        }

        const char* btype = block ? llmJsonStr(block, "type") : NULL;
        if (btype != NULL && strcmp(btype, "tool_use") == 0)
        {
            llmEmit(sink, LLMAPI_STREAM_TOOL_CALL, NULL,
                    llmJsonStr(block, "id"), llmJsonStr(block, "name"),
                    index);
        }
        else if (btype != NULL && strcmp(btype, "server_tool_use") == 0)
        {
            const char* name = llmJsonStr(block, "name");
            llmEmitServerTool(sink, name, "started", index);
            llmSinkRememberServerTool(sink, index, name);
        }
    }
    else if (strcmp(type, "content_block_delta") == 0)
    {
        cJSON* delta = cJSON_GetObjectItem(root, "delta");
        size_t index = 0;
        cJSON* idx = cJSON_GetObjectItem(root, "index");
        if (idx != NULL && cJSON_IsNumber(idx))
        {
            index = (size_t)idx->valuedouble;
        }

        const char* dtype = delta ? llmJsonStr(delta, "type") : NULL;
        if (dtype == NULL)
        {
            // nothing
        }
        else if (strcmp(dtype, "text_delta") == 0)
        {
            llmEmit(sink, LLMAPI_STREAM_TEXT, llmJsonStr(delta, "text"),
                    NULL, NULL, index);
        }
        else if (strcmp(dtype, "thinking_delta") == 0)
        {
            llmEmit(sink, LLMAPI_STREAM_THINKING,
                    llmJsonStr(delta, "thinking"), NULL, NULL, index);
        }
        else if (strcmp(dtype, "input_json_delta") == 0)
        {
            // Server-side tools stream their input too; don't surface it as a
            // client tool call.
            if (llmSinkFindServerTool(sink, index) == NULL)
            {
                llmEmit(sink, LLMAPI_STREAM_TOOL_CALL,
                        llmJsonStr(delta, "partial_json"), NULL, NULL, index);
            }
        }
        else if (strcmp(dtype, "citations_delta") == 0)
        {
            cJSON* cit = cJSON_GetObjectItem(delta, "citation");
            if (cit != NULL)
            {
                LlmmapiCitation c;
                memset(&c, 0, sizeof(c));
                c.url        = llmJsonStr(cit, "url");
                c.title      = llmJsonStr(cit, "title");
                c.citedText  = llmJsonStr(cit, "cited_text");
                c.startIndex = -1;
                c.endIndex   = -1;
                llmJsonLong(cit, "start_char_index", &c.startIndex);
                llmJsonLong(cit, "end_char_index", &c.endIndex);

                if (c.url != NULL)
                {
                    llmEmitCitation(sink, &c);
                }
            }
        }
    }
    else if (strcmp(type, "content_block_stop") == 0)
    {
        size_t index = 0;
        cJSON* idx = cJSON_GetObjectItem(root, "index");
        if (idx != NULL && cJSON_IsNumber(idx))
        {
            index = (size_t)idx->valuedouble;
        }

        const char* name = llmSinkFindServerTool(sink, index);
        if (name != NULL)
        {
            llmEmitServerTool(sink, name, "completed", index);
        }
    }
    else if (strcmp(type, "message_delta") == 0)
    {
        cJSON* usage = cJSON_GetObjectItem(root, "usage");
        long outTok = 0;
        if (usage != NULL && llmJsonLong(usage, "output_tokens", &outTok))
        {
            sink->outputTokens = outTok;
        }
    }
    else if (strcmp(type, "message_stop") == 0)
    {
        sink->done = 1;
        llmEmit(sink, LLMAPI_STREAM_DONE, NULL, NULL, NULL, 0);
    }

    cJSON_Delete(root);
}

// ==================== Provider: OpenAI Responses ====================

static void responsesAddInputPart(cJSON* content, const LlmmapiPart* part)
{
    if (part->type == LLMMAPI_PART_TEXT)
    {
        cJSON* b = cJSON_CreateObject();
        if (b == NULL) return;
        cJSON_AddStringToObject(b, "type", "input_text");
        cJSON_AddStringToObject(b, "text", part->text ? part->text : "");
        cJSON_AddItemToArray(content, b);
    }
    else if (part->type == LLMMAPI_PART_IMAGE)
    {
        cJSON* b = cJSON_CreateObject();
        if (b == NULL) return;
        cJSON_AddStringToObject(b, "type", "input_image");
        if (part->fileId != NULL)
        {
            cJSON_AddStringToObject(b, "file_id", part->fileId);
        }
        else
        {
            cJSON_AddStringToObject(b, "image_url",
                                    part->url ? part->url : "");
        }
        cJSON_AddItemToArray(content, b);
    }
    else if (part->type == LLMMAPI_PART_FILE)
    {
        cJSON* b = cJSON_CreateObject();
        if (b == NULL) return;
        cJSON_AddStringToObject(b, "type", "input_file");
        if (part->fileId != NULL)
        {
            cJSON_AddStringToObject(b, "file_id", part->fileId);
        }
        else
        {
            if (part->filename != NULL)
            {
                cJSON_AddStringToObject(b, "filename", part->filename);
            }
            if (part->url != NULL)
            {
                cJSON_AddStringToObject(b, "file_data", part->url);
            }
        }
        cJSON_AddItemToArray(content, b);
    }
}

static int responsesEncode(LlmmapiCtx* ctx, const LlmmapiReq* req, int stream,
                           LlmBuf* body)
{
    (void)ctx;

    cJSON* root = cJSON_CreateObject();
    if (root == NULL)
    {
        return -1;
    }

    cJSON_AddStringToObject(root, "model", req->model);

    // System messages become the top-level "instructions" field.
    LlmBuf instructions;
    llmBufInit(&instructions);
    for (size_t i = 0; i < req->messageCount; i++)
    {
        const LlmmapiMessage* msg = req->messages[i];
        if (strcmp(msg->role, LLMAPI_ROLE_SYSTEM) != 0)
        {
            continue;
        }

        for (size_t j = 0; j < msg->partCount; j++)
        {
            if (msg->parts[j].type == LLMMAPI_PART_TEXT &&
                msg->parts[j].text != NULL)
            {
                if (instructions.len > 0)
                {
                    llmBufAppend(&instructions, "\n\n");
                }
                llmBufAppend(&instructions, msg->parts[j].text);
            }
        }
    }
    if (instructions.len > 0)
    {
        cJSON_AddStringToObject(root, "instructions", instructions.data);
    }
    llmBufFree(&instructions);

    cJSON* input = cJSON_AddArrayToObject(root, "input");
    if (input == NULL)
    {
        cJSON_Delete(root);
        return -1;
    }

    for (size_t i = 0; i < req->messageCount; i++)
    {
        const LlmmapiMessage* msg = req->messages[i];

        if (strcmp(msg->role, LLMAPI_ROLE_SYSTEM) == 0)
        {
            continue;
        }

        // Provider-native output items (e.g. web_search_call) are replayed
        // as top-level input items for the Responses API.
        int hadRawBlock = 0;
        int hasNonRawPart = 0;
        for (size_t j = 0; j < msg->partCount; j++)
        {
            const LlmmapiPart* part = &msg->parts[j];
            if (part->type == LLMMAPI_PART_RAW_BLOCK)
            {
                if (part->provider != NULL && part->rawJson != NULL &&
                    strcmp(part->provider, "openai-responses") == 0)
                {
                    cJSON* obj = cJSON_Parse(part->rawJson);
                    if (obj != NULL && cJSON_IsObject(obj))
                    {
                        cJSON_AddItemToArray(input, obj);
                    }
                    else
                    {
                        cJSON_Delete(obj);
                    }
                }
                hadRawBlock = 1;
            }
            else
            {
                hasNonRawPart = 1;
            }
        }

        if (hadRawBlock && !hasNonRawPart)
        {
            continue;
        }

        // Tool results are top-level function_call_output items.
        if (strcmp(msg->role, LLMAPI_ROLE_TOOL) == 0 ||
            llmMessageHasPart(msg, LLMMAPI_PART_TOOL_RESULT))
        {
            for (size_t j = 0; j < msg->partCount; j++)
            {
                const LlmmapiPart* part = &msg->parts[j];
                if (part->type != LLMMAPI_PART_TOOL_RESULT)
                {
                    continue;
                }

                cJSON* item = cJSON_CreateObject();
                if (item == NULL) continue;
                cJSON_AddStringToObject(item, "type", "function_call_output");
                cJSON_AddStringToObject(item, "call_id",
                                        part->toolCallId ? part->toolCallId
                                                         : "");
                cJSON_AddStringToObject(item, "output",
                                        part->text ? part->text : "");
                cJSON_AddItemToArray(input, item);
            }
            continue;
        }

        // Assistant tool calls are top-level function_call items.
        if (llmMessageHasPart(msg, LLMMAPI_PART_TOOL_CALL))
        {
            char* text = llmConcatParts(msg, LLMMAPI_PART_TEXT);
            if (text != NULL && text[0] != '\0')
            {
                cJSON* item = cJSON_CreateObject();
                if (item != NULL)
                {
                    cJSON_AddStringToObject(item, "role", "assistant");
                    cJSON* content = cJSON_AddArrayToObject(item, "content");
                    if (content != NULL)
                    {
                        cJSON* b = cJSON_CreateObject();
                        if (b != NULL)
                        {
                            cJSON_AddStringToObject(b, "type", "input_text");
                            cJSON_AddStringToObject(b, "text", text);
                            cJSON_AddItemToArray(content, b);
                        }
                    }
                    cJSON_AddItemToArray(input, item);
                }
            }
            free(text);

            for (size_t j = 0; j < msg->partCount; j++)
            {
                const LlmmapiPart* part = &msg->parts[j];
                if (part->type != LLMMAPI_PART_TOOL_CALL)
                {
                    continue;
                }

                cJSON* item = cJSON_CreateObject();
                if (item == NULL) continue;
                cJSON_AddStringToObject(item, "type", "function_call");
                cJSON_AddStringToObject(item, "call_id",
                                        part->toolCallId ? part->toolCallId
                                                         : "");
                cJSON_AddStringToObject(item, "name",
                                        part->toolName ? part->toolName : "");
                cJSON_AddStringToObject(item, "arguments",
                                        part->text ? part->text : "{}");
                cJSON_AddItemToArray(input, item);
            }
            continue;
        }

        // Ordinary message.
        cJSON* item = cJSON_CreateObject();
        if (item == NULL) continue;

        const char* role = (strcmp(msg->role, LLMAPI_ROLE_ASSISTANT) == 0)
                               ? "assistant"
                               : "user";
        cJSON_AddStringToObject(item, "role", role);

        int hasMultipart = 0;
        for (size_t j = 0; j < msg->partCount; j++)
        {
            LlmmapiPartType t = msg->parts[j].type;
            if (t == LLMMAPI_PART_IMAGE || t == LLMMAPI_PART_FILE)
            {
                hasMultipart = 1;
                break;
            }
        }

        if (!hasMultipart)
        {
            char* text = llmConcatParts(msg, LLMMAPI_PART_TEXT);
            cJSON_AddStringToObject(item, "content", text ? text : "");
            free(text);
        }
        else
        {
            cJSON* content = cJSON_AddArrayToObject(item, "content");
            if (content != NULL)
            {
                for (size_t j = 0; j < msg->partCount; j++)
                {
                    responsesAddInputPart(content, &msg->parts[j]);
                }
            }
        }

        cJSON_AddItemToArray(input, item);
    }

    // Tools use the flat Responses schema (no nested "function" object).
    if (req->toolCount > 0 || req->rawToolCount > 0)
    {
        cJSON* tools = cJSON_AddArrayToObject(root, "tools");
        if (tools != NULL)
        {
            for (size_t i = 0; i < req->toolCount; i++)
            {
                const LlmTool* tool = &req->tools[i];

                cJSON* t = cJSON_CreateObject();
                if (t == NULL) continue;
                cJSON_AddStringToObject(t, "type", "function");
                cJSON_AddStringToObject(t, "name",
                                        tool->name ? tool->name : "");
                if (tool->description != NULL)
                {
                    cJSON_AddStringToObject(t, "description",
                                            tool->description);
                }

                cJSON* schema = tool->parametersJson
                                    ? cJSON_Parse(tool->parametersJson)
                                    : NULL;
                if (schema == NULL)
                {
                    schema = cJSON_CreateObject();
                }
                if (schema != NULL)
                {
                    cJSON_AddItemToObject(t, "parameters", schema);
                }

                cJSON_AddItemToArray(tools, t);
            }

            llmAppendRawTools(req, tools);
        }
    }

    if (req->hasMaxTokens)
    {
        cJSON_AddNumberToObject(root, "max_output_tokens",
                                (double)req->maxTokens);
    }
    if (req->hasTemperature)
    {
        cJSON_AddNumberToObject(root, "temperature", req->temperature);
    }
    if (req->hasTopP)
    {
        cJSON_AddNumberToObject(root, "top_p", req->topP);
    }

    // Note: the Responses API has no "stop" parameter; llmapiReqSetStop() is
    // intentionally ignored for this provider.

    cJSON_AddBoolToObject(root, "stream", stream ? 1 : 0);

    for (size_t i = 0; i < req->extraCount; i++)
    {
        llmAddJsonValue(root, req->extraKeys[i], req->extraValues[i]);
    }

    char* json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (json == NULL)
    {
        return -1;
    }

    int rc = llmBufAppend(body, json);
    free(json);

    return rc;
}

static int responsesDecode(LlmmapiCtx* ctx, const char* rawJson,
                           LlmmapiResp* out)
{
    cJSON* root = cJSON_Parse(rawJson);
    if (root == NULL)
    {
        return -1;
    }

    out->rawJson = llmStrDup(rawJson);

    cJSON* output = cJSON_GetObjectItem(root, "output");
    if (output != NULL && cJSON_IsArray(output))
    {
        cJSON* item = NULL;
        cJSON_ArrayForEach(item, output)
        {
            const char* type = llmJsonStr(item, "type");
            if (type == NULL)
            {
                continue;
            }

            if (strcmp(type, "message") == 0)
            {
                cJSON* content = cJSON_GetObjectItem(item, "content");
                if (content != NULL && cJSON_IsArray(content))
                {
                    cJSON* part = NULL;
                    cJSON_ArrayForEach(part, content)
                    {
                        const char* ptype = llmJsonStr(part, "type");
                        if (ptype != NULL &&
                            strcmp(ptype, "output_text") == 0)
                        {
                            llmAddTextPart(&out->message, LLMMAPI_PART_TEXT,
                                           llmJsonStr(part, "text"));

                            cJSON* annotations =
                                cJSON_GetObjectItem(part, "annotations");
                            if (annotations != NULL &&
                                cJSON_IsArray(annotations))
                            {
                                cJSON* a = NULL;
                                cJSON_ArrayForEach(a, annotations)
                                {
                                    const char* atype =
                                        llmJsonStr(a, "type");
                                    if (atype == NULL ||
                                        strcmp(atype, "url_citation") != 0)
                                    {
                                        continue;
                                    }

                                    const char* url = llmJsonStr(a, "url");
                                    if (url == NULL)
                                    {
                                        continue;
                                    }

                                    LlmmapiCitation* c =
                                        llmRespAddCitation(out);
                                    if (c == NULL)
                                    {
                                        continue;
                                    }

                                    c->url        = llmStrDup(url);
                                    c->title      = llmStrDup(
                                        llmJsonStr(a, "title"));
                                    c->startIndex = -1;
                                    c->endIndex   = -1;
                                    llmJsonLong(a, "start_index",
                                                &c->startIndex);
                                    llmJsonLong(a, "end_index",
                                                &c->endIndex);
                                }
                            }
                        }
                    }
                }
            }
            else if (strcmp(type, "function_call") == 0)
            {
                LlmmapiPart* p = llmMessageNewPart(&out->message);
                if (p == NULL) continue;

                p->type       = LLMMAPI_PART_TOOL_CALL;
                p->toolCallId = llmStrDup(llmJsonStr(item, "call_id"));
                p->toolName   = llmStrDup(llmJsonStr(item, "name"));
                p->text       = llmStrDup(llmJsonStr(item, "arguments"));
                if (p->text == NULL)
                {
                    p->text = llmStrDup("{}");
                }
            }
            else if (strcmp(type, "reasoning") == 0)
            {
                cJSON* summary = cJSON_GetObjectItem(item, "summary");
                if (summary != NULL && cJSON_IsArray(summary))
                {
                    cJSON* s = NULL;
                    cJSON_ArrayForEach(s, summary)
                    {
                        llmAddTextPart(&out->message, LLMMAPI_PART_THINKING,
                                       llmJsonStr(s, "text"));
                    }
                }

                cJSON* content = cJSON_GetObjectItem(item, "content");
                if (content != NULL && cJSON_IsArray(content))
                {
                    cJSON* c = NULL;
                    cJSON_ArrayForEach(c, content)
                    {
                        llmAddTextPart(&out->message, LLMMAPI_PART_THINKING,
                                       llmJsonStr(c, "text"));
                    }
                }
            }
            else
            {
                // Preserve provider-native output items (web_search_call, ...)
                // so callers can replay them.
                char* raw = cJSON_PrintUnformatted(item);
                if (raw != NULL)
                {
                    LlmmapiPart* p = llmMessageNewPart(&out->message);
                    if (p != NULL)
                    {
                        p->type     = LLMMAPI_PART_RAW_BLOCK;
                        p->provider = llmStrDup("openai-responses");
                        p->rawJson  = raw;
                        raw = NULL;
                    }
                    free(raw);
                }
            }
        }
    }

    const char* status = llmJsonStr(root, "status");
    if (status != NULL)
    {
        out->finishReason = llmStrDup(status);
    }

    cJSON* incomplete = cJSON_GetObjectItem(root, "incomplete_details");
    if (incomplete != NULL)
    {
        const char* reason = llmJsonStr(incomplete, "reason");
        if (reason != NULL)
        {
            free(out->finishReason);
            out->finishReason = llmStrDup(reason);
        }
    }

    cJSON* usage = cJSON_GetObjectItem(root, "usage");
    long in = 0, outTok = 0, total = 0;
    if (usage != NULL)
    {
        llmJsonLong(usage, "input_tokens", &in);
        llmJsonLong(usage, "output_tokens", &outTok);
        if (!llmJsonLong(usage, "total_tokens", &total))
        {
            total = in + outTok;
        }
    }

    out->inputTokens  = in;
    out->outputTokens = outTok;
    out->totalTokens  = total;

    cJSON* error = cJSON_GetObjectItem(root, "error");
    if (error != NULL && !cJSON_IsNull(error))
    {
        const char* etype   = llmJsonStr(error, "type");
        const char* message = llmJsonStr(error, "message");

        if (etype != NULL && message != NULL)
        {
            llmBufAppendFmt(&ctx->errbuf, "%s: %s", etype, message);
        }
        else if (message != NULL)
        {
            llmBufAppend(&ctx->errbuf, message);
        }
        else if (etype != NULL)
        {
            llmBufAppend(&ctx->errbuf, etype);
        }
        else
        {
            llmBufAppend(&ctx->errbuf, "Responses API error");
        }

        cJSON_Delete(root);
        return -1;
    }

    cJSON_Delete(root);

    return 0;
}

static void responsesStreamDecode(const char* payload, LlmStreamSink* sink)
{
    cJSON* root = cJSON_Parse(payload);
    if (root == NULL)
    {
        return;
    }

    const char* type = llmJsonStr(root, "type");
    if (type == NULL)
    {
        cJSON_Delete(root);
        return;
    }

    size_t index = 0;
    cJSON* oi = cJSON_GetObjectItem(root, "output_index");
    if (oi != NULL && cJSON_IsNumber(oi))
    {
        index = (size_t)oi->valuedouble;
    }

    if (strcmp(type, "response.output_item.added") == 0)
    {
        cJSON* item = cJSON_GetObjectItem(root, "item");
        const char* itype = item ? llmJsonStr(item, "type") : NULL;
        if (itype != NULL && strcmp(itype, "function_call") == 0)
        {
            llmEmit(sink, LLMAPI_STREAM_TOOL_CALL, NULL,
                    llmJsonStr(item, "call_id"), llmJsonStr(item, "name"),
                    index);
        }
        else if (itype != NULL && strcmp(itype, "web_search_call") == 0)
        {
            llmEmitServerTool(sink, "web_search", "started", index);
        }
    }
    else if (strcmp(type, "response.output_text.delta") == 0)
    {
        llmEmit(sink, LLMAPI_STREAM_TEXT, llmJsonStr(root, "delta"),
                NULL, NULL, index);
    }
    else if (strcmp(type, "response.function_call_arguments.delta") == 0)
    {
        llmEmit(sink, LLMAPI_STREAM_TOOL_CALL, llmJsonStr(root, "delta"),
                NULL, NULL, index);
    }
    else if (strcmp(type, "response.reasoning_summary_text.delta") == 0 ||
             strcmp(type, "response.reasoning_text.delta") == 0)
    {
        llmEmit(sink, LLMAPI_STREAM_THINKING, llmJsonStr(root, "delta"),
                NULL, NULL, index);
    }
    else if (strcmp(type, "response.output_text.annotation.added") == 0)
    {
        cJSON* a = cJSON_GetObjectItem(root, "annotation");
        if (a != NULL)
        {
            const char* atype = llmJsonStr(a, "type");
            if (atype != NULL && strcmp(atype, "url_citation") == 0)
            {
                const char* url = llmJsonStr(a, "url");
                if (url != NULL)
                {
                    LlmmapiCitation c;
                    memset(&c, 0, sizeof(c));
                    c.url        = url;
                    c.title      = llmJsonStr(a, "title");
                    c.startIndex = -1;
                    c.endIndex   = -1;
                    llmJsonLong(a, "start_index", &c.startIndex);
                    llmJsonLong(a, "end_index", &c.endIndex);
                    llmEmitCitation(sink, &c);
                }
            }
        }
    }
    else if (strcmp(type, "response.web_search_call.in_progress") == 0)
    {
        llmEmitServerTool(sink, "web_search", "in_progress", index);
    }
    else if (strcmp(type, "response.web_search_call.searching") == 0)
    {
        llmEmitServerTool(sink, "web_search", "searching", index);
    }
    else if (strcmp(type, "response.web_search_call.completed") == 0)
    {
        llmEmitServerTool(sink, "web_search", "completed", index);
    }
    else if (strcmp(type, "response.completed") == 0 ||
             strcmp(type, "response.incomplete") == 0 ||
             strcmp(type, "response.failed") == 0)
    {
        cJSON* response = cJSON_GetObjectItem(root, "response");
        cJSON* usage = response ? cJSON_GetObjectItem(response, "usage") : NULL;
        if (usage != NULL)
        {
            long in = 0, outTok = 0;
            if (llmJsonLong(usage, "input_tokens", &in))
            {
                sink->inputTokens = in;
            }
            if (llmJsonLong(usage, "output_tokens", &outTok))
            {
                sink->outputTokens = outTok;
            }
        }

        // Fallback: some providers only carry annotations in the final payload.
        if (!sink->hadCitation && response != NULL)
        {
            cJSON* output = cJSON_GetObjectItem(response, "output");
            if (output != NULL && cJSON_IsArray(output))
            {
                cJSON* oitem = NULL;
                cJSON_ArrayForEach(oitem, output)
                {
                    const char* otype = llmJsonStr(oitem, "type");
                    if (otype == NULL || strcmp(otype, "message") != 0)
                    {
                        continue;
                    }

                    cJSON* content = cJSON_GetObjectItem(oitem, "content");
                    if (content == NULL || !cJSON_IsArray(content))
                    {
                        continue;
                    }

                    cJSON* part = NULL;
                    cJSON_ArrayForEach(part, content)
                    {
                        const char* ptype = llmJsonStr(part, "type");
                        if (ptype == NULL ||
                            strcmp(ptype, "output_text") != 0)
                        {
                            continue;
                        }

                        llmEmitCitationArray(
                            cJSON_GetObjectItem(part, "annotations"), sink);
                    }
                }
            }
        }

        sink->done = 1;
        llmEmit(sink, LLMAPI_STREAM_DONE, NULL, NULL, NULL, 0);
    }
    else if (strcmp(type, "error") == 0)
    {
        sink->done = 1;
        llmEmit(sink, LLMAPI_STREAM_DONE, NULL, NULL, NULL, 0);
    }

    cJSON_Delete(root);
}

// ==================== Provider table ====================

static const char* const gAnthropicStaticHeaders[] = {
    "anthropic-version: " LLMAPI_ANTHROPIC_VERSION,
    NULL
};

static const LlmProvider gOpenAIProvider = {
    "/chat/completions",
    "Authorization",
    "Bearer ",
    NULL,
    openaiEncode,
    openaiDecode,
    openaiStreamDecode
};

static const LlmProvider gAnthropicProvider = {
    "/messages",
    "x-api-key",
    "",
    gAnthropicStaticHeaders,
    anthropicEncode,
    anthropicDecode,
    anthropicStreamDecode
};

static const LlmProvider gResponsesProvider = {
    "/responses",
    "Authorization",
    "Bearer ",
    NULL,
    responsesEncode,
    responsesDecode,
    responsesStreamDecode
};

static const LlmProvider* llmGetProvider(const LlmmapiCtx* ctx)
{
    switch (ctx->api)
    {
        case LLMMAPI_API_ANTHROPIC:
            return &gAnthropicProvider;
        case LLMMAPI_API_OPENAI_RESPONSES:
            return &gResponsesProvider;
        default:
            return &gOpenAIProvider;
    }
}

// ==================== Internal: URL / headers ====================

static int llmJoinUrl(LlmBuf* out, const char* baseUrl, const char* path)
{
    if (baseUrl == NULL || path == NULL)
    {
        return -1;
    }

    if (llmBufAppend(out, baseUrl) != 0)
    {
        return -1;
    }

    if (path[0] == '\0' || out->len == 0)
    {
        return llmBufAppend(out, path);
    }

    int baseSlash = (out->data[out->len - 1] == '/');
    int pathSlash = (path[0] == '/');

    if (baseSlash && pathSlash)
    {
        return llmBufAppend(out, path + 1);
    }

    if (!baseSlash && !pathSlash)
    {
        if (llmBufAppend(out, "/") != 0)
        {
            return -1;
        }
    }

    return llmBufAppend(out, path);
}

static int llmBuildHeaders(LlmmapiCtx* ctx, const LlmProvider* provider,
                           const char* apiKey, int stream, LlmBuf* authBuf,
                           const char** headers, size_t headersCap)
{
    if (llmHasCrlf(apiKey))
    {
        llmBufAppend(&ctx->errbuf, "apiKey contains invalid characters");
        return -1;
    }

    size_t n = 0;

    if (n + 1 >= headersCap) return -1;
    headers[n++] = "Content-Type: application/json";

    if (stream)
    {
        if (n + 1 >= headersCap) return -1;
        headers[n++] = "Accept: text/event-stream";
    }

    if (apiKey != NULL)
    {
        if (llmBufAppend(authBuf, provider->authHeader) != 0 ||
            llmBufAppend(authBuf, ": ") != 0 ||
            llmBufAppend(authBuf, provider->authPrefix) != 0 ||
            llmBufAppend(authBuf, apiKey) != 0)
        {
            llmBufAppend(&ctx->errbuf, "Out of memory");
            return -1;
        }

        if (n + 1 >= headersCap) return -1;
        headers[n++] = authBuf->data;
    }

    if (provider->staticHeaders != NULL)
    {
        for (size_t i = 0; provider->staticHeaders[i] != NULL; i++)
        {
            if (n + 1 >= headersCap) return -1;
            headers[n++] = provider->staticHeaders[i];
        }
    }

    headers[n] = NULL;

    return 0;
}

// ==================== Internal: chunk collector ====================

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

// ==================== Sync completion ====================

LlmmapiResp* llmapiChatComplete(LlmmapiCtx* ctx,
                                const char* baseUrl,
                                const char* apiKey,
                                const LlmmapiReq* req)
{
    if (ctx == NULL || baseUrl == NULL || req == NULL)
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

    if (llmHasCrlf(baseUrl))
    {
        llmBufAppend(&ctx->errbuf, "baseUrl contains invalid characters");
        return NULL;
    }

    const LlmProvider* provider = llmGetProvider(ctx);

    LlmBuf bodyBuf;
    llmBufInit(&bodyBuf);

    if (provider->encode(ctx, req, 0, &bodyBuf) != 0)
    {
        if (ctx->errbuf.len == 0)
        {
            llmBufAppend(&ctx->errbuf, "Failed to build request body");
        }
        llmBufFree(&bodyBuf);
        return NULL;
    }

    LlmBuf urlBuf;
    LlmBuf authBuf;
    llmBufInit(&urlBuf);
    llmBufInit(&authBuf);

    if (llmJoinUrl(&urlBuf, baseUrl, provider->path) != 0)
    {
        llmBufAppend(&ctx->errbuf, "Failed to build request URL");
        llmBufFree(&bodyBuf);
        llmBufFree(&urlBuf);
        return NULL;
    }

    const char* headers[8];
    if (llmBuildHeaders(ctx, provider, apiKey, 0, &authBuf, headers, 8) != 0)
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
                     (result.error != NULL) ? result.error : "transport error");
        llmBufFree(&respBuf);
        return NULL;
    }

    if (result.status < 200 || result.status >= 300)
    {
        llmBufAppendFmt(&ctx->errbuf, "HTTP %ld: %s", result.status,
                        respBuf.data ? respBuf.data : "");
        llmBufFree(&respBuf);
        return NULL;
    }

    LlmmapiResp* resp = (LlmmapiResp*)calloc(1, sizeof(LlmmapiResp));
    if (resp == NULL)
    {
        llmBufAppend(&ctx->errbuf, "Out of memory");
        llmBufFree(&respBuf);
        return NULL;
    }

    llmMessageInit(&resp->message);
    resp->message.role = llmStrDup("assistant");

    if (provider->decode(ctx, respBuf.data ? respBuf.data : "", resp) != 0)
    {
        if (ctx->errbuf.len == 0)
        {
            llmBufAppend(&ctx->errbuf, "Failed to parse response");
        }
        llmBufFree(&respBuf);
        llmapiRespFree(resp);
        return NULL;
    }

    llmBufFree(&respBuf);

    return resp;
}

// ==================== Streaming ====================

struct LlmStreamCtx
{
    LlmBuf             lineBuf;
    LlmBuf             rawBuf;
    const LlmProvider* provider;
    LlmStreamSink      sink;
};

static void llmProcessSseLine(struct LlmStreamCtx* sctx, const char* line,
                              size_t len)
{
    if (sctx->sink.done || len == 0)
    {
        return;
    }

    // Accept both "data:" and "data: ".  Anthropic "event:" lines are ignored.
    if (len < 5 || strncmp(line, "data:", 5) != 0)
    {
        return;
    }

    const char* payload = line + 5;
    if (*payload == ' ')
    {
        payload++;
    }

    sctx->provider->streamDecode(payload, &sctx->sink);
}

static int llmStreamWriteCallback(const void* data, size_t total,
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
            llmProcessSseLine(sctx, sctx->lineBuf.data, sctx->lineBuf.len);

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

int llmapiChatCompleteStream(LlmmapiCtx* ctx,
                             const char* baseUrl,
                             const char* apiKey,
                             const LlmmapiReq* req,
                             LlmmapiStreamCb cb,
                             void* userdata)
{
    if (ctx == NULL || baseUrl == NULL || req == NULL || cb == NULL)
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

    if (llmHasCrlf(baseUrl))
    {
        llmBufAppend(&ctx->errbuf, "baseUrl contains invalid characters");
        return -1;
    }

    const LlmProvider* provider = llmGetProvider(ctx);

    LlmBuf bodyBuf;
    llmBufInit(&bodyBuf);

    if (provider->encode(ctx, req, 1, &bodyBuf) != 0)
    {
        if (ctx->errbuf.len == 0)
        {
            llmBufAppend(&ctx->errbuf, "Failed to build request body");
        }
        llmBufFree(&bodyBuf);
        return -1;
    }

    LlmBuf urlBuf;
    LlmBuf authBuf;
    llmBufInit(&urlBuf);
    llmBufInit(&authBuf);

    if (llmJoinUrl(&urlBuf, baseUrl, provider->path) != 0)
    {
        llmBufAppend(&ctx->errbuf, "Failed to build request URL");
        llmBufFree(&bodyBuf);
        llmBufFree(&urlBuf);
        return -1;
    }

    const char* headers[8];
    if (llmBuildHeaders(ctx, provider, apiKey, 1, &authBuf, headers, 8) != 0)
    {
        llmBufFree(&bodyBuf);
        llmBufFree(&urlBuf);
        llmBufSecureFree(&authBuf);
        return -1;
    }

    struct LlmStreamCtx sctx;
    llmBufInit(&sctx.lineBuf);
    llmBufInit(&sctx.rawBuf);
    sctx.provider = provider;
    sctx.sink.cb         = cb;
    sctx.sink.userdata   = userdata;
    sctx.sink.done       = 0;
    sctx.sink.inputTokens  = 0;
    sctx.sink.outputTokens = 0;
    sctx.sink.hadCitation  = 0;
    sctx.sink.serverToolCount = 0;

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
    llmSinkFreeServerTools(&sctx.sink);

    ctx->lastPromptTokens     = sctx.sink.inputTokens;
    ctx->lastCompletionTokens = sctx.sink.outputTokens;
    ctx->lastTotalTokens      = sctx.sink.inputTokens + sctx.sink.outputTokens;

    if (rc != 0 || (result.error != NULL && result.error[0] != '\0'))
    {
        llmBufAppend(&ctx->errbuf,
                     (result.error != NULL) ? result.error : "transport error");
        llmBufFree(&sctx.rawBuf);
        return -1;
    }

    if (result.status < 200 || result.status >= 300)
    {
        llmBufAppendFmt(&ctx->errbuf, "HTTP %ld: %s", result.status,
                        sctx.rawBuf.data ? sctx.rawBuf.data : "");
        llmBufFree(&sctx.rawBuf);
        return -1;
    }

    llmBufFree(&sctx.rawBuf);

    if (!sctx.sink.done)
    {
        sctx.sink.done = 1;
        llmEmit(&sctx.sink, LLMAPI_STREAM_DONE, NULL, NULL, NULL, 0);
    }

    return 0;
}

void llmapiStreamGetUsage(LlmmapiCtx* ctx,
                          long* total, long* input, long* output)
{
    if (ctx == NULL)
    {
        if (total)  *total = 0;
        if (input)  *input = 0;
        if (output) *output = 0;
        return;
    }

    if (total)  *total  = ctx->lastTotalTokens;
    if (input)  *input  = ctx->lastPromptTokens;
    if (output) *output = ctx->lastCompletionTokens;
}

// ==================== Response accessors ====================

const LlmmapiMessage* llmapiRespGetMessage(const LlmmapiResp* resp)
{
    return resp ? &resp->message : NULL;
}

char* llmapiRespGetText(const LlmmapiResp* resp)
{
    return resp ? llmConcatParts(&resp->message, LLMMAPI_PART_TEXT) : NULL;
}

char* llmapiRespGetThinking(const LlmmapiResp* resp)
{
    return resp ? llmConcatParts(&resp->message, LLMMAPI_PART_THINKING) : NULL;
}

const char* llmapiRespGetFinishReason(const LlmmapiResp* resp)
{
    return resp ? resp->finishReason : NULL;
}

void llmapiRespGetUsage(const LlmmapiResp* resp,
                        long* input, long* output, long* total)
{
    if (resp == NULL)
    {
        if (input)  *input = 0;
        if (output) *output = 0;
        if (total)  *total = 0;
        return;
    }

    if (input)  *input  = resp->inputTokens;
    if (output) *output = resp->outputTokens;
    if (total)  *total  = resp->totalTokens;
}

const char* llmapiRespGetRawJson(const LlmmapiResp* resp)
{
    return resp ? resp->rawJson : NULL;
}

size_t llmapiRespGetCitationCount(const LlmmapiResp* resp)
{
    return resp ? resp->citationCount : 0;
}

const LlmmapiCitation* llmapiRespGetCitation(const LlmmapiResp* resp,
                                             size_t idx)
{
    if (resp == NULL || idx >= resp->citationCount)
    {
        return NULL;
    }

    return &resp->citations[idx];
}

void llmapiRespFree(LlmmapiResp* resp)
{
    if (resp == NULL)
    {
        return;
    }

    llmMessageFree(&resp->message);
    free(resp->finishReason);
    free(resp->rawJson);

    for (size_t i = 0; i < resp->citationCount; i++)
    {
        free((void*)resp->citations[i].url);
        free((void*)resp->citations[i].title);
        free((void*)resp->citations[i].citedText);
    }
    free(resp->citations);

    free(resp);
}

// ==================== Raw request ====================

char* llmapiRequest(LlmmapiCtx* ctx,
                    const char* baseUrl,
                    const char* apiKey,
                    const char* method,
                    const char* path,
                    const char* contentType,
                    const void* body,
                    size_t bodyLen)
{
    if (ctx == NULL || baseUrl == NULL || method == NULL || path == NULL)
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

    if (method[0] == '\0' || llmHasCrlf(method))
    {
        llmBufAppend(&ctx->errbuf, "invalid method");
        return NULL;
    }

    if (llmHasCrlf(baseUrl) || llmHasCrlf(path) ||
        llmHasCrlf(apiKey) || llmHasCrlf(contentType))
    {
        llmBufAppend(&ctx->errbuf, "invalid characters in request fields");
        return NULL;
    }

    if (body == NULL && bodyLen != 0)
    {
        llmBufAppend(&ctx->errbuf, "body is NULL but bodyLen is non-zero");
        return NULL;
    }

    const LlmProvider* provider = llmGetProvider(ctx);

    LlmBuf urlBuf;
    LlmBuf authBuf;
    LlmBuf ctBuf;
    llmBufInit(&urlBuf);
    llmBufInit(&authBuf);
    llmBufInit(&ctBuf);

    if (llmJoinUrl(&urlBuf, baseUrl, path) != 0)
    {
        llmBufAppend(&ctx->errbuf, "Failed to build request URL");
        llmBufFree(&urlBuf);
        return NULL;
    }

    const char* headers[8];
    size_t nh = 0;

    if (contentType != NULL)
    {
        if (llmBufAppend(&ctBuf, "Content-Type: ") != 0 ||
            llmBufAppend(&ctBuf, contentType) != 0)
        {
            llmBufAppend(&ctx->errbuf, "Out of memory");
            llmBufFree(&urlBuf);
            llmBufFree(&ctBuf);
            return NULL;
        }
        headers[nh++] = ctBuf.data;
    }

    if (apiKey != NULL)
    {
        if (llmBufAppend(&authBuf, provider->authHeader) != 0 ||
            llmBufAppend(&authBuf, ": ") != 0 ||
            llmBufAppend(&authBuf, provider->authPrefix) != 0 ||
            llmBufAppend(&authBuf, apiKey) != 0)
        {
            llmBufAppend(&ctx->errbuf, "Out of memory");
            llmBufFree(&urlBuf);
            llmBufFree(&ctBuf);
            llmBufFree(&authBuf);
            return NULL;
        }
        headers[nh++] = authBuf.data;
    }

    if (provider->staticHeaders != NULL)
    {
        for (size_t i = 0; provider->staticHeaders[i] != NULL && nh + 1 < 8; i++)
        {
            headers[nh++] = provider->staticHeaders[i];
        }
    }

    headers[nh] = NULL;

    LlmmapiHttpRequest httpReq;
    httpReq.method    = method;
    httpReq.url       = urlBuf.data;
    httpReq.headers   = headers;
    httpReq.body      = body;
    httpReq.bodyLen   = bodyLen;
    httpReq.timeoutMs = 120000L;

    LlmBuf respBuf;
    llmBufInit(&respBuf);

    LlmmapiHttpResult result;
    result.status = 0;
    result.error  = NULL;

    int rc = ctx->transport.perform(ctx->transport.userdata, &httpReq,
                                    llmCollectChunk, &respBuf, &result);

    llmBufFree(&urlBuf);
    llmBufFree(&ctBuf);
    llmBufSecureFree(&authBuf);

    if (rc != 0 || (result.error != NULL && result.error[0] != '\0'))
    {
        llmBufAppend(&ctx->errbuf,
                     (result.error != NULL) ? result.error : "transport error");
        llmBufFree(&respBuf);
        return NULL;
    }

    if (result.status < 200 || result.status >= 300)
    {
        llmBufAppendFmt(&ctx->errbuf, "HTTP %ld: %s", result.status,
                        respBuf.data ? respBuf.data : "");
        llmBufFree(&respBuf);
        return NULL;
    }

    return llmBufDetach(&respBuf);
}
