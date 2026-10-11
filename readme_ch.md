# libllmapi

使用 **cjson** 调用 OpenAI 兼容 LLM API 的 C99 库。

HTTP(S) 传输层**不内置**，由调用方注入，因此不绑定任何 HTTP/TLS 实现（libcurl、WinHTTP、esp_http_client、mock 等）。

支持普通文本、推理思考（thinking）、工具调用（tool calls）的同步/流式输出。

## 依赖

- [cjson](https://github.com/DaveGamble/cJSON)

## 编译

```sh
cc -std=c99 -c llmapi.c -o llmapi.o
cc -std=c99 your_program.c llmapi.o -lcjson -o your_program
```

## 使用示例

### 注入传输层（必需）

库本身不做任何网络 I/O，发请求前必须先注入一个 transport：

```c
static int myPerform(void* ud, const LlmmapiHttpRequest* req,
                     LlmmapiWriteFn onChunk, void* cud,
                     LlmmapiHttpResult* res)
{
    // 在这里驱动 libcurl / WinHTTP / esp_http_client 等。
    // 每收到一段响应体就调用：onChunk(data, len, cud);
    // 然后回填：
    res->status = 200;   // HTTP 状态码
    res->error  = NULL;  // 传输失败时填错误串
    return 0;            // 完整往返返回 0，失败返回非 0
}

LlmmapiTransport t = { myPerform, NULL, NULL }; // perform, destroy, userdata
llmapiSetTransport(ctx, &t);
```

### 消息内容

`content` 是自由文本：换行、制表符、非 ASCII（UTF-8）都可以。库会负责做 JSON 转义，
所以多行的 system prompt、多行的用户文本都能原样送达。

只有会进入 **HTTP 请求行或请求头**的取值才做 CRLF 注入校验：

| 取值 | 是否校验 | 原因 |
|------|----------|------|
| `baseUrl` | 是 | 会拼成请求行（追加 `/chat/completions` 之后） |
| `apiKey` | 是 | 会拼成 `Authorization: Bearer <key>` |
| 消息 `role` | 是 | 必须是普通标识符 |
| 消息 `content` | **否** | 进的是 JSON 请求体，正文里的换行是合法的 |

非法参数会被**静默忽略**：指针为 NULL，或 `role` 含 CRLF，都会直接丢掉这条消息而不报错。
如果发出去的请求里消息条数比你添加的少，先检查 role。

### 普通调用

```c
LlmmapiCtx* ctx = llmapiInit();

LlmmapiTransport t = { myPerform, NULL, NULL };
llmapiSetTransport(ctx, &t);

LlmmapiReq* req = llmapiReqNew("gpt-4");
llmapiReqAddMsg(req, LLMAPI_ROLE_SYSTEM, "你是一个有用的助手。");
llmapiReqAddMsg(req, LLMAPI_ROLE_USER,   "法国的首都是哪里？");
llmapiReqSetTemperature(req, 0.7);
llmapiReqSetMaxTokens(req, 200);

char* resp = llmapiChatComplete(ctx, "https://api.openai.com/v1", "sk-...", req);
if (resp)
{
    char* text = llmapiRespGetContent(resp, 0);
    printf("回复: %s\n", text);
    free(text);
    printf("Tokens: %d\n", llmapiRespGetTotalTokens(resp));
    free(resp);
}
else
{
    fprintf(stderr, "错误: %s\n", llmapiStrError(ctx));
}

llmapiReqFree(req);
llmapiCleanup(ctx);
```

### 流式调用（区分文本、思维链、工具调用）

```c
void onChunk(LlmmapiStreamType type, const char* content, int done, void* ud)
{
    switch (type)
    {
        case LLMAPI_STREAM_TEXT:      printf("%s", content);              break;
        case LLMAPI_STREAM_THINKING:  printf("[思考] %s", content);       break;
        case LLMAPI_STREAM_TOOL_CALL: printf("[工具] %s", content);       break;
    }
    if (done) printf("\n[完成]\n");
}

llmapiChatCompleteStream(ctx, url, key, req, onChunk, NULL);
```

### 连续对话

```c
char* resp = llmapiChatComplete(ctx, url, key, req);
char* text = llmapiRespGetContent(resp, 0);
llmapiReqAddMsg(req, LLMAPI_ROLE_ASSISTANT, text);  // 追加模型回复
free(text);
free(resp);

llmapiReqAddMsg(req, LLMAPI_ROLE_USER, "下一个问题");
resp = llmapiChatComplete(ctx, url, key, req);       // 带完整历史
```

### 工具调用（Tool Calling）

```c
// 提出需要工具的请求
llmapiReqAddMsg(req, LLMAPI_ROLE_USER, "巴黎今天天气如何？");
char* resp = llmapiChatComplete(ctx, url, key, req);

// 解析工具调用请求
char* tcJson = llmapiRespGetToolCallsJson(resp, 0);
free(resp);

// 追加 assistant 的 tool_calls 消息
llmapiReqAddAssistantToolCalls(req, tcJson);

// 执行工具并追加结果
llmapiReqAddToolResult(req, "call_123", "get_weather", "晴朗，22°C");

// 继续对话
resp = llmapiChatComplete(ctx, url, key, req);
char* answer = llmapiRespGetContent(resp, 0);
```

### 额外参数

```c
llmapiReqSet(req, "top_p",            "0.9");
llmapiReqSet(req, "presence_penalty", "0.1");
llmapiReqSet(req, "stop",             "[\"END\"]");
llmapiReqSet(req, "seed",             "42");
```

## API 一览

### 生命周期 / 错误

| 函数 | 说明 |
|------|------|
| `llmapiInit()` / `llmapiCleanup()` | 生命周期 |
| `llmapiStrError(ctx)` | 错误信息 |

### 传输层

| 函数 | 说明 |
|------|------|
| `llmapiSetTransport(ctx, &t)` | 注入 HTTP 传输层（调用前必需） |
| `LlmmapiTransport` | `{ perform, destroy, userdata }` 回调结构体 |
| `LlmmapiHttpRequest` / `LlmmapiHttpResult` | 传给传输层的请求 / 由传输层回填的结果 |
| `LlmmapiWriteFn` | 响应体分块回调 |

### 请求

| 函数 | 说明 |
|------|------|
| `llmapiReqNew(model)` / `llmapiReqFree(req)` | 请求生命周期 |
| `llmapiReqAddMsg(req, role, content)` | 添加消息 (role: system/user/assistant/tool)，content 可多行 |
| `llmapiReqAddAssistantToolCalls(req, json)` | 添加带 tool_calls 的 assistant 消息 |
| `llmapiReqAddToolResult(req, id, name, content)` | 添加工具执行结果消息 |
| `llmapiReqSetMaxTokens(req, n)` | 设置 max_tokens |
| `llmapiReqSetTemperature(req, t)` | 设置 temperature |
| `llmapiReqSet(req, key, jsonValue)` | 通用参数 |

### 调用

| 函数 | 说明 |
|------|------|
| `llmapiChatComplete(ctx, url, key, req)` | 普通调用，返回 JSON 字符串 |
| `llmapiChatCompleteStream(ctx, url, key, req, cb, ud)` | 流式调用，类型感知回调 |
| `llmapiStreamGetUsage(ctx, &total, &prompt, &comp)` | 获取上一次流式调用的 token 用量 |

### 响应提取

| 函数 | 说明 |
|------|------|
| `llmapiRespGetContent(json, idx)` | 提取 `choices[idx].message.content` |
| `llmapiRespGetReasoningContent(json, idx)` | 提取 `choices[idx].message.reasoning_content` |
| `llmapiRespGetToolCallsJson(json, idx)` | 序列化 `choices[idx].message.tool_calls` |
| `llmapiRespGetTotalTokens(json)` | `usage.total_tokens` |
| `llmapiRespGetPromptTokens(json)` | `usage.prompt_tokens` |
| `llmapiRespGetCompletionTokens(json)` | `usage.completion_tokens` |

### 流式回调类型

| 值 | 含义 |
|----|------|
| `LLMAPI_STREAM_TEXT` | 普通回复文本 |
| `LLMAPI_STREAM_THINKING` | 思维链 / 思考过程 (DeepSeek-R1 等) |
| `LLMAPI_STREAM_TOOL_CALL` | 工具调用增量 (JSON 字符串) |

## 协议

BSD 2-Clause。
