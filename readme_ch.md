# libllmapi

使用**与 provider 无关的消息模型**调用 LLM API 的 C99 库。

- 协议：**OpenAI Chat Completions** (`/chat/completions`)、
  **OpenAI Responses** (`/responses`) 与 **Anthropic** (`/messages`)
- HTTP(S) 传输层**不内置**，由调用方注入（libcurl、WinHTTP、esp_http_client、mock 等）
- JSON 使用 **cjson**
- 支持同步/流式、文本/思考/工具调用、图片与文件

## 依赖

- [cjson](https://github.com/DaveGamble/cJSON)

## 编译

```sh
cc -std=c99 -c llmapi.c -o llmapi.o
cc -std=c99 your_program.c llmapi.o -lcjson -o your_program
```

## 使用

### 1. 注入传输层（必需）

库本身不做任何网络 I/O，发请求前必须先注入一个 transport：

```c
static int myPerform(void* ud, const LlmmapiHttpRequest* req,
                     LlmmapiWriteFn onChunk, void* cud,
                     LlmmapiHttpResult* res)
{
    // 在这里驱动 libcurl / WinHTTP / esp_http_client 等。
    // 每收到一段响应体就调用：onChunk(data, len, cud);
    res->status = 200;   // HTTP 状态码
    res->error  = NULL;  // 传输失败时填错误串
    return 0;            // 完整往返返回 0，失败返回非 0
}

LlmmapiTransport t = { myPerform, NULL, NULL };
llmapiSetTransport(ctx, &t);
```

### 2. 选择协议

```c
llmapiSetApi(ctx, LLMMAPI_API_OPENAI);           // 默认（chat completions）
llmapiSetApi(ctx, LLMMAPI_API_OPENAI_RESPONSES); // /responses
llmapiSetApi(ctx, LLMMAPI_API_ANTHROPIC);        // /messages
```

Anthropic 的 `baseUrl` 要带版本，例如 `https://api.anthropic.com/v1`
（provider 会自动追加 `/messages`）。Responses 用
`https://api.openai.com/v1`（会自动追加 `/responses`）。

`LLMMAPI_API_OPENAI_RESPONSES` 的映射：
- system 消息 → 顶层 `instructions`
- `max_tokens` → `max_output_tokens`
- 工具调用/结果 → `function_call` / `function_call_output` items
- `llmapiReqSetStop()` 会被忽略（Responses API 没有 `stop`）

### 3. 普通调用

```c
LlmmapiCtx* ctx = llmapiInit();
LlmmapiTransport t = { myPerform, NULL, NULL };
llmapiSetTransport(ctx, &t);
llmapiSetApi(ctx, LLMMAPI_API_OPENAI);

LlmmapiReq* req = llmapiReqNew("gpt-4");
llmapiReqAddMsg(req, LLMAPI_ROLE_SYSTEM, "你是一个有用的助手。");
llmapiReqAddMsg(req, LLMAPI_ROLE_USER,   "法国的首都是哪里？");
llmapiReqSetMaxTokens(req, 200);
llmapiReqSetTemperature(req, 0.7);

LlmmapiResp* resp = llmapiChatComplete(ctx, "https://api.openai.com/v1",
                                       "sk-...", req);
if (resp)
{
    char* text = llmapiRespGetText(resp);
    printf("回复: %s\n", text);
    free(text);

    long in, out, total;
    llmapiRespGetUsage(resp, &in, &out, &total);
    printf("Tokens: %ld\n", total);

    llmapiRespFree(resp);
}
else
{
    fprintf(stderr, "错误: %s\n", llmapiStrError(ctx));
}

llmapiReqFree(req);
llmapiCleanup(ctx);
```

### 4. 流式调用

```c
static void onEvent(const LlmmapiStreamEvent* ev, void* ud)
{
    switch (ev->type)
    {
        case LLMAPI_STREAM_TEXT:
            printf("%s", ev->text ? ev->text : "");
            break;
        case LLMAPI_STREAM_THINKING:
            printf("[思考] %s", ev->text ? ev->text : "");
            break;
        case LLMAPI_STREAM_TOOL_CALL:
            printf("[工具 %zu %s] %s", ev->toolIndex,
                   ev->toolName ? ev->toolName : "",
                   ev->text ? ev->text : "");
            break;
        case LLMAPI_STREAM_DONE:
            printf("\n[完成]\n");
            break;
    }
}

llmapiChatCompleteStream(ctx, url, key, req, onEvent, NULL);

long total, in, out;
llmapiStreamGetUsage(ctx, &total, &in, &out);
```

工具参数是增量 JSON：把同一个 `ev->toolIndex` 的 `ev->text` 依次拼接即可。

### 5. 多模态（图片 / 文件）

库**不读文件、不做 base64**。在应用层生成 `data:` URL，再作为一个 part 加入：

```c
LlmmapiMessage* m = llmapiReqAddMessage(req, LLMAPI_ROLE_USER);
llmapiMsgAddText(m, "图里是什么？");
llmapiMsgAddImage(m, "data:image/png;base64,iVBORw0KG...");   // 调用方编码

// PDF / 二进制文档（内联 data URL）
llmapiMsgAddFile(m, "doc.pdf", "application/pdf",
                 "data:application/pdf;base64,JVBERi0...");
```

纯文本文件无需特殊处理——读成字符串后用 `llmapiReqAddMsg` 发送（支持多行）。

### 6. 工具调用

```c
// 定义工具（JSON Schema）
llmapiReqAddTool(req, "get_weather", "查询天气",
                 "{\"type\":\"object\",\"properties\":{"
                 "\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}");

LlmmapiResp* resp = llmapiChatComplete(ctx, url, key, req);

// 遍历响应 parts
const LlmmapiMessage* msg = llmapiRespGetMessage(resp);
for (size_t i = 0; i < llmapiMessageGetPartCount(msg); i++)
{
    const LlmmapiPart* p = llmapiMessageGetPart(msg, i);
    if (p->type == LLMMAPI_PART_TOOL_CALL)
    {
        printf("调用 %s %s(%s)\n", p->toolCallId, p->toolName, p->text);

        // 追加 assistant 的 tool call
        LlmmapiMessage* a = llmapiReqAddMessage(req, LLMAPI_ROLE_ASSISTANT);
        llmapiMsgAddToolCall(a, p->toolCallId, p->toolName, p->text);

        // 追加工具结果
        LlmmapiMessage* t = llmapiReqAddMessage(req, LLMAPI_ROLE_TOOL);
        llmapiMsgAddToolResult(t, p->toolCallId, "晴朗，22°C");
    }
}

llmapiRespFree(resp);

// 带上工具结果继续问
resp = llmapiChatComplete(ctx, url, key, req);
```

provider 会自动翻译：
- OpenAI：`tool_calls` / role `tool`
- Anthropic：`tool_use` / `tool_result` content blocks

### 7. 原始请求

通过注入的 transport 发送任意字节（使用当前 provider 的鉴权方式）：

```c
char* resp = llmapiRequest(ctx, "https://api.openai.com/v1", "sk-...",
                           "POST", "/files",
                           "multipart/form-data; boundary=----xyz",
                           multipartBody, multipartLen);
```

会把整个响应缓冲后返回，不适合流式。

## API 一览

### 生命周期 / 错误 / 传输 / 协议

| 函数 | 说明 |
|------|------|
| `llmapiInit()` / `llmapiCleanup()` | 生命周期 |
| `llmapiStrError(ctx)` | 错误信息 |
| `llmapiSetTransport(ctx, &t)` | 注入 HTTP 传输层（必需） |
| `llmapiSetApi(ctx, api)` | 选择 `LLMMAPI_API_OPENAI` / `LLMMAPI_API_OPENAI_RESPONSES` / `LLMMAPI_API_ANTHROPIC` |

### 中立模型

| 类型 / 函数 | 说明 |
|-------------|------|
| `LlmmapiPart` / `LlmmapiPartType` | 文本、图片、文件、思考、工具调用、工具结果 |
| `LlmmapiMessage` | role + parts |
| `llmapiMessageGetPartCount(msg)` | part 数量 |
| `llmapiMessageGetPart(msg, idx)` | 借用的 part 指针 |

### 请求

| 函数 | 说明 |
|------|------|
| `llmapiReqNew(model)` / `llmapiReqFree(req)` | 请求生命周期 |
| `llmapiReqAddMessage(req, role)` | 添加消息，返回可写句柄 |
| `llmapiReqAddMsg(req, role, text)` | 便捷：加一条单 TEXT 消息 |
| `llmapiMsgAddText(msg, text)` | 添加 TEXT |
| `llmapiMsgAddImage(msg, url)` | 添加 IMAGE（http(s) 或 data: URL） |
| `llmapiMsgAddFile(msg, filename, mime, dataUrlOrFileId)` | 添加 FILE |
| `llmapiMsgAddToolCall(msg, id, name, argsJson)` | 添加 TOOL_CALL |
| `llmapiMsgAddToolResult(msg, id, content)` | 添加 TOOL_RESULT |
| `llmapiReqAddTool(req, name, description, schemaJson)` | 定义工具 |
| `llmapiReqSetMaxTokens` / `SetTemperature` / `SetTopP` | 生成参数 |
| `llmapiReqSetStop(req, jsonArray)` | 停止序列 |
| `llmapiReqSetExtra(req, key, json)` | provider 原生顶层参数 |

### 调用

| 函数 | 说明 |
|------|------|
| `llmapiChatComplete(...)` | 同步调用，返回 `LlmmapiResp*` |
| `llmapiChatCompleteStream(..., cb, ud)` | 流式调用，中立事件 |
| `llmapiStreamGetUsage(ctx, &total, &in, &out)` | 上一次流式调用的用量 |

### 响应

| 函数 | 说明 |
|------|------|
| `llmapiRespGetMessage(resp)` | 借用的 assistant 消息 |
| `llmapiRespGetText(resp)` | 拼接后的 TEXT（malloc） |
| `llmapiRespGetThinking(resp)` | 拼接后的 THINKING（malloc） |
| `llmapiRespGetFinishReason(resp)` | 借用的结束原因 |
| `llmapiRespGetUsage(resp, &in, &out, &total)` | token 用量 |
| `llmapiRespGetRawJson(resp)` | 借用的原始响应 |
| `llmapiRespFree(resp)` | 释放响应 |

### 原始请求

| 函数 | 说明 |
|------|------|
| `llmapiRequest(ctx, url, key, method, path, contentType, body, len)` | 发送任意字节，返回缓冲后的响应体 |

## 协议

BSD 2-Clause。
