# libllmapi

A C99 library for calling LLM APIs (OpenAI-compatible) using **cjson**.

The HTTP(S) transport is **not built in** — you inject one, so the library works
with any HTTP/TLS stack (libcurl, WinHTTP, esp_http_client, a mock, ...).

## Dependencies

- [cjson](https://github.com/DaveGamble/cJSON)

## Build

```sh
cc -std=c99 -c llmapi.c -o llmapi.o
cc -std=c99 your_program.c llmapi.o -lcjson -o your_program
```

## Usage

### Transport (required)

libllmapi performs no network I/O itself. Implement a transport and inject it
before making any call:

```c
static int myPerform(void* ud, const LlmmapiHttpRequest* req,
                     LlmmapiWriteFn onChunk, void* cud,
                     LlmmapiHttpResult* res)
{
    // e.g. drive libcurl / WinHTTP / esp_http_client here.
    // For each chunk of the response body: onChunk(data, len, cud);
    // Then set:
    res->status = 200;   // HTTP status code
    res->error  = NULL;  // non-NULL string on transport failure
    return 0;            // 0 on a completed round-trip, non-zero on error
}

LlmmapiTransport t = { myPerform, NULL, NULL }; // perform, destroy, userdata
llmapiSetTransport(ctx, &t);
```

### Message content

`content` is free-form text. Newlines, tabs and non-ASCII characters (UTF-8) are
all fine — the library JSON-escapes them for you, so a multi-line system prompt
or a multi-line user message round-trips unchanged.

Only values that end up in the **HTTP request line or headers** are validated
against CRLF injection:

| Value | Validated | Reason |
|-------|-----------|--------|
| `baseUrl` | yes | becomes the request line (after appending `/chat/completions`) |
| `apiKey` | yes | becomes `Authorization: Bearer <key>` |
| message `role` | yes | must be a plain token |
| message `content` | **no** | goes into the JSON body, where newlines are legal |

Invalid arguments are ignored **silently**: a NULL pointer, or a `role`
containing CRLF, drops the message without setting an error. If a request comes
back with fewer messages than you added, check the roles first.

### Sync Chat

```c
LlmmapiCtx* ctx = llmapiInit();

LlmmapiTransport t = { myPerform, NULL, NULL };
llmapiSetTransport(ctx, &t);

LlmmapiReq* req = llmapiReqNew("gpt-4");
llmapiReqAddMsg(req, LLMAPI_ROLE_SYSTEM, "You are a helpful assistant.");
llmapiReqAddMsg(req, LLMAPI_ROLE_USER,   "What is the capital of France?");
llmapiReqSetTemperature(req, 0.7);
llmapiReqSetMaxTokens(req, 200);

char* resp = llmapiChatComplete(ctx, "https://api.openai.com/v1", "sk-...", req);
if (resp)
{
    char* text = llmapiRespGetContent(resp, 0);
    printf("Reply: %s\n", text);
    free(text);
    printf("Tokens: %d\n", llmapiRespGetTotalTokens(resp));
    free(resp);
}
else
{
    fprintf(stderr, "Error: %s\n", llmapiStrError(ctx));
}

llmapiReqFree(req);
llmapiCleanup(ctx);
```

### Streaming

```c
void onChunk(LlmmapiStreamType type, const char* content, int done, void* ud)
{
    switch (type)
    {
        case LLMAPI_STREAM_TEXT:     printf("%s", content); break;
        case LLMAPI_STREAM_THINKING: printf("[think] %s", content); break;
        case LLMAPI_STREAM_TOOL_CALL:printf("[tool] %s", content); break;
    }
    if (done) printf("\n[DONE]\n");
}

llmapiChatCompleteStream(ctx, "https://api.openai.com/v1", "sk-...", req, onChunk, NULL);
```

### Continuing a Conversation

```c
// Keep the same req and add messages to it across turns
char* resp = llmapiChatComplete(ctx, url, key, req);
char* text = llmapiRespGetContent(resp, 0);
llmapiReqAddMsg(req, LLMAPI_ROLE_ASSISTANT, text);  // append model reply
free(text);
free(resp);

llmapiReqAddMsg(req, LLMAPI_ROLE_USER, "next question");
resp = llmapiChatComplete(ctx, url, key, req);       // full history sent
```

### Tool Calling

```c
// Ask a question that needs a tool
llmapiReqAddMsg(req, LLMAPI_ROLE_USER, "What's the weather in Paris?");
char* resp = llmapiChatComplete(ctx, url, key, req);

// Read the tool call request
char* tcJson = llmapiRespGetToolCallsJson(resp, 0);
free(resp);

// Append assistant tool_calls message
llmapiReqAddAssistantToolCalls(req, tcJson);

// Execute the tool and append result
llmapiReqAddToolResult(req, "call_123", "get_weather", "Sunny, 22°C");

// Continue
resp = llmapiChatComplete(ctx, url, key, req);
char* answer = llmapiRespGetContent(resp, 0);
```

### Extra Parameters

```c
llmapiReqSet(req, "top_p",            "0.9");
llmapiReqSet(req, "presence_penalty", "0.1");
llmapiReqSet(req, "stop",             "[\"END\"]");
llmapiReqSet(req, "seed",             "42");
```

### Multimodal Input (raw content parts)

`llmapiReqAddMsgJson` accepts the `content` field as raw JSON. The library does
**not** read files or base64-encode anything — you build the array yourself
(e.g. with a base64 encoder in your app):

```c
// caller-provided data: URL (base64 produced outside the library)
const char* parts =
    "[{\"type\":\"text\",\"text\":\"What is in this image?\"},"
    " {\"type\":\"image_url\","
    "  \"image_url\":{\"url\":\"data:image/png;base64,iVBORw0KG...\"}}]";

llmapiReqAddMsgJson(req, LLMAPI_ROLE_USER, parts);
char* resp = llmapiChatComplete(ctx, url, key, req);
```

Text files need no special handling — read them into a string and pass to
`llmapiReqAddMsg` (multi-line content is supported).

### Raw Request

`llmapiRequest` sends arbitrary bytes through the injected transport. It sets
only `Content-Type` (when given) and `Authorization: Bearer <apiKey>` (when
`apiKey` is non-NULL):

```c
// e.g. a caller-built multipart upload to the Files API
char* resp = llmapiRequest(ctx, "https://api.openai.com/v1", "sk-...",
                           "POST", "/files",
                           "multipart/form-data; boundary=----xyz",
                           multipartBody, multipartLen);
```

> This buffers the whole response. For chat, prefer the higher-level helpers.

## API

### Lifecycle / Error

| Function | Description |
|----------|-------------|
| `llmapiInit()` / `llmapiCleanup()` | Lifecycle |
| `llmapiStrError(ctx)` | Last error message |

### Transport

| Function | Description |
|----------|-------------|
| `llmapiSetTransport(ctx, &t)` | Inject the HTTP transport (required before any call) |
| `LlmmapiTransport` | `{ perform, destroy, userdata }` callback struct |
| `LlmmapiHttpRequest` / `LlmmapiHttpResult` | Request passed in / result filled out by the transport |
| `LlmmapiWriteFn` | Chunk callback for response body bytes |

### Request

| Function | Description |
|----------|-------------|
| `llmapiReqNew(model)` / `llmapiReqFree(req)` | Request lifecycle |
| `llmapiReqAddMsg(req, role, content)` | Add a message (role: system/user/assistant/tool); `content` may be multi-line |
| `llmapiReqAddMsgJson(req, role, json)` | Add a message with raw JSON `content` parts (no encoding done) |
| `llmapiReqAddAssistantToolCalls(req, json)` | Add assistant message with tool_calls |
| `llmapiReqAddToolResult(req, id, name, content)` | Add tool result message |
| `llmapiReqSetMaxTokens(req, n)` | Set max_tokens |
| `llmapiReqSetTemperature(req, t)` | Set temperature |
| `llmapiReqSet(req, key, jsonValue)` | Generic parameter |

### Chat

| Function | Description |
|----------|-------------|
| `llmapiChatComplete(ctx, url, key, req)` | Sync (returns JSON string) |
| `llmapiChatCompleteStream(ctx, url, key, req, cb, ud)` | Streaming with type-aware callback |
| `llmapiStreamGetUsage(ctx, &total, &prompt, &comp)` | Token usage from last streaming call |

### Raw Request

| Function | Description |
|----------|-------------|
| `llmapiRequest(ctx, url, key, method, path, contentType, body, len)` | Send raw bytes via the transport; returns the buffered response body |

### Response Helpers

| Function | Description |
|----------|-------------|
| `llmapiRespGetContent(json, idx)` | `choices[idx].message.content` |
| `llmapiRespGetReasoningContent(json, idx)` | `choices[idx].message.reasoning_content` |
| `llmapiRespGetToolCallsJson(json, idx)` | Serialized `choices[idx].message.tool_calls` JSON |
| `llmapiRespGetTotalTokens(json)` | `usage.total_tokens` |
| `llmapiRespGetPromptTokens(json)` | `usage.prompt_tokens` |
| `llmapiRespGetCompletionTokens(json)` | `usage.completion_tokens` |

### Streaming Types

| Value | Meaning |
|-------|---------|
| `LLMAPI_STREAM_TEXT` | Regular output text |
| `LLMAPI_STREAM_THINKING` | Reasoning / chain-of-thought (DeepSeek-R1, etc) |
| `LLMAPI_STREAM_TOOL_CALL` | Tool call delta (JSON string) |

## License

BSD 2-Clause. See LICENSE file for details.
