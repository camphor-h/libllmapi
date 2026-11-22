# libllmapi

A C99 library for calling LLM APIs with a **provider-neutral message model**.

- Wire protocols: **OpenAI Chat Completions** (`/chat/completions`),
  **OpenAI Responses** (`/responses`) and **Anthropic** (`/messages`)
- The HTTP(S) transport is **not built in** — you inject one (libcurl, WinHTTP,
  esp_http_client, a mock, ...)
- JSON is handled with **cjson**
- Sync and streaming, text / thinking / tool calls, images and files

## Dependencies

- [cjson](https://github.com/DaveGamble/cJSON)

## Build

```sh
cc -std=c99 -c llmapi.c -o llmapi.o
cc -std=c99 your_program.c llmapi.o -lcjson -o your_program
```

## Usage

### 1. Transport (required)

libllmapi performs no network I/O itself. Implement and inject a transport:

```c
static int myPerform(void* ud, const LlmmapiHttpRequest* req,
                     LlmmapiWriteFn onChunk, void* cud,
                     LlmmapiHttpResult* res)
{
    // Drive libcurl / WinHTTP / esp_http_client here.
    // For each response body chunk: onChunk(data, len, cud);
    res->status = 200;   // HTTP status code
    res->error  = NULL;  // non-NULL string on transport failure
    return 0;            // 0 on a completed round-trip, non-zero on error
}

LlmmapiTransport t = { myPerform, NULL, NULL };
llmapiSetTransport(ctx, &t);
```

### 2. Select a provider

```c
llmapiSetApi(ctx, LLMMAPI_API_OPENAI);           // default (chat completions)
llmapiSetApi(ctx, LLMMAPI_API_OPENAI_RESPONSES); // /responses
llmapiSetApi(ctx, LLMMAPI_API_ANTHROPIC);        // /messages
```

For Anthropic, pass a `baseUrl` that includes the version, e.g.
`https://api.anthropic.com/v1` (the provider appends `/messages`).
For Responses, `https://api.openai.com/v1` (the provider appends `/responses`).

Notes for `LLMMAPI_API_OPENAI_RESPONSES`:
- system messages map to the top-level `instructions`
- `max_tokens` maps to `max_output_tokens`
- tool calls/results map to `function_call` / `function_call_output` items
- `llmapiReqSetStop()` is ignored (the Responses API has no `stop`)

### 3. Sync chat

```c
LlmmapiCtx* ctx = llmapiInit();
LlmmapiTransport t = { myPerform, NULL, NULL };
llmapiSetTransport(ctx, &t);
llmapiSetApi(ctx, LLMMAPI_API_OPENAI);

LlmmapiReq* req = llmapiReqNew("gpt-4");
llmapiReqAddMsg(req, LLMAPI_ROLE_SYSTEM, "You are a helpful assistant.");
llmapiReqAddMsg(req, LLMAPI_ROLE_USER,   "What is the capital of France?");
llmapiReqSetMaxTokens(req, 200);
llmapiReqSetTemperature(req, 0.7);

LlmmapiResp* resp = llmapiChatComplete(ctx, "https://api.openai.com/v1",
                                       "sk-...", req);
if (resp)
{
    char* text = llmapiRespGetText(resp);
    printf("Reply: %s\n", text);
    free(text);

    long in, out, total;
    llmapiRespGetUsage(resp, &in, &out, &total);
    printf("Tokens: %ld\n", total);

    llmapiRespFree(resp);
}
else
{
    fprintf(stderr, "Error: %s\n", llmapiStrError(ctx));
}

llmapiReqFree(req);
llmapiCleanup(ctx);
```

### 4. Streaming

```c
static void onEvent(const LlmmapiStreamEvent* ev, void* ud)
{
    switch (ev->type)
    {
        case LLMAPI_STREAM_TEXT:
            printf("%s", ev->text ? ev->text : "");
            break;
        case LLMAPI_STREAM_THINKING:
            printf("[think] %s", ev->text ? ev->text : "");
            break;
        case LLMAPI_STREAM_TOOL_CALL:
            printf("[tool %zu %s] %s", ev->toolIndex,
                   ev->toolName ? ev->toolName : "",
                   ev->text ? ev->text : "");
            break;
        case LLMAPI_STREAM_DONE:
            printf("\n[DONE]\n");
            break;
    }
}

llmapiChatCompleteStream(ctx, url, key, req, onEvent, NULL);

long total, in, out;
llmapiStreamGetUsage(ctx, &total, &in, &out);
```

Tool-call arguments arrive as incremental JSON: append `ev->text` for the same
`ev->toolIndex` until done.

### 5. Multimodal (images / files)

The library does **not** read files or base64-encode anything. Build a `data:`
URL in your app and add it as a part:

```c
LlmmapiMessage* m = llmapiReqAddMessage(req, LLMAPI_ROLE_USER);
llmapiMsgAddText(m, "What is in this image?");
llmapiMsgAddImage(m, "data:image/png;base64,iVBORw0KG...");   // caller-encoded

// PDF / binary document (inline data URL)
llmapiMsgAddFile(m, "doc.pdf", "application/pdf",
                 "data:application/pdf;base64,JVBERi0...");
```

Plain text files need no special handling — read them into a string and use
`llmapiReqAddMsg` (multi-line text is supported).

### 6. Tool calling

```c
// Define a tool (JSON Schema)
llmapiReqAddTool(req, "get_weather", "Get the weather",
                 "{\"type\":\"object\",\"properties\":{"
                 "\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}");

LlmmapiResp* resp = llmapiChatComplete(ctx, url, key, req);

// Walk response parts
const LlmmapiMessage* msg = llmapiRespGetMessage(resp);
for (size_t i = 0; i < llmapiMessageGetPartCount(msg); i++)
{
    const LlmmapiPart* p = llmapiMessageGetPart(msg, i);
    if (p->type == LLMMAPI_PART_TOOL_CALL)
    {
        printf("call %s %s(%s)\n", p->toolCallId, p->toolName, p->text);

        // Append the assistant tool call to the history
        LlmmapiMessage* a = llmapiReqAddMessage(req, LLMAPI_ROLE_ASSISTANT);
        llmapiMsgAddToolCall(a, p->toolCallId, p->toolName, p->text);

        // Append the tool result
        LlmmapiMessage* t = llmapiReqAddMessage(req, LLMAPI_ROLE_TOOL);
        llmapiMsgAddToolResult(t, p->toolCallId, "Sunny, 22C");
    }
}

llmapiRespFree(resp);

// Ask again with the tool result in history
resp = llmapiChatComplete(ctx, url, key, req);
```

The provider translates:
- OpenAI: `tool_calls` / role `tool`
- Anthropic: `tool_use` / `tool_result` content blocks

### 7. Raw request

Send arbitrary bytes through the injected transport (uses the selected
provider's auth scheme):

```c
char* resp = llmapiRequest(ctx, "https://api.openai.com/v1", "sk-...",
                           "POST", "/files",
                           "multipart/form-data; boundary=----xyz",
                           multipartBody, multipartLen);
```

Buffers the whole response; not for streaming.

## API

### Lifecycle / Error / Transport / Provider

| Function | Description |
|----------|-------------|
| `llmapiInit()` / `llmapiCleanup()` | Lifecycle |
| `llmapiStrError(ctx)` | Last error message |
| `llmapiSetTransport(ctx, &t)` | Inject the HTTP transport (required) |
| `llmapiSetApi(ctx, api)` | Select `LLMMAPI_API_OPENAI` / `LLMMAPI_API_OPENAI_RESPONSES` / `LLMMAPI_API_ANTHROPIC` |

### Neutral model

| Type / Function | Description |
|-----------------|-------------|
| `LlmmapiPart` / `LlmmapiPartType` | Text, image, file, thinking, tool call, tool result |
| `LlmmapiMessage` | role + parts |
| `llmapiMessageGetPartCount(msg)` | Number of parts |
| `llmapiMessageGetPart(msg, idx)` | Borrowed part pointer |

### Request

| Function | Description |
|----------|-------------|
| `llmapiReqNew(model)` / `llmapiReqFree(req)` | Request lifecycle |
| `llmapiReqAddMessage(req, role)` | Add a message, returns a mutable handle |
| `llmapiReqAddMsg(req, role, text)` | Convenience: one TEXT part |
| `llmapiMsgAddText(msg, text)` | Add TEXT |
| `llmapiMsgAddImage(msg, url)` | Add IMAGE (http(s) or data: URL) |
| `llmapiMsgAddFile(msg, filename, mime, dataUrlOrFileId)` | Add FILE |
| `llmapiMsgAddToolCall(msg, id, name, argsJson)` | Add TOOL_CALL |
| `llmapiMsgAddToolResult(msg, id, content)` | Add TOOL_RESULT |
| `llmapiReqAddTool(req, name, description, schemaJson)` | Define a tool |
| `llmapiReqSetMaxTokens` / `SetTemperature` / `SetTopP` | Generation params |
| `llmapiReqSetStop(req, jsonArray)` | Stop sequences |
| `llmapiReqSetExtra(req, key, json)` | Provider-native top-level param |

### Completion

| Function | Description |
|----------|-------------|
| `llmapiChatComplete(...)` | Sync; returns `LlmmapiResp*` |
| `llmapiChatCompleteStream(..., cb, ud)` | Streaming; neutral events |
| `llmapiStreamGetUsage(ctx, &total, &in, &out)` | Usage of the last stream |

### Response

| Function | Description |
|----------|-------------|
| `llmapiRespGetMessage(resp)` | Borrowed assistant message |
| `llmapiRespGetText(resp)` | Concatenated TEXT (malloc'd) |
| `llmapiRespGetThinking(resp)` | Concatenated THINKING (malloc'd) |
| `llmapiRespGetFinishReason(resp)` | Borrowed finish/stop reason |
| `llmapiRespGetUsage(resp, &in, &out, &total)` | Token usage |
| `llmapiRespGetRawJson(resp)` | Borrowed raw provider response |
| `llmapiRespFree(resp)` | Free a response |

### Raw Request

| Function | Description |
|----------|-------------|
| `llmapiRequest(ctx, url, key, method, path, contentType, body, len)` | Send raw bytes; returns buffered body |

## License

BSD 2-Clause. See LICENSE file for details.
