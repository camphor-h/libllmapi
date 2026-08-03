# libllmapi

使用 **libcurl** + **cjson** 调用 OpenAI 兼容 LLM API 的 C99 库。

支持普通文本、推理思考（thinking）、工具调用（tool calls）的同步/流式输出。

## 依赖

- [libcurl](https://curl.se/libcurl/)
- [cjson](https://github.com/DaveGamble/cJSON)

## 编译

```sh
cc -std=c99 -c llmapi.c -o llmapi.o
cc -std=c99 your_program.c llmapi.o -lcurl -lcjson -o your_program
```

## 使用示例

### 普通调用

```c
LlmmapiCtx* ctx = llmapiInit();

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

### 生命周期 / 错误 / 配置

| 函数 | 说明 |
|------|------|
| `llmapiInit()` / `llmapiCleanup()` | 生命周期 |
| `llmapiStrError(ctx)` | 错误信息 |
| `llmapiSetCaPath(ctx, path)` | 自定义 CA 证书路径 |

### 请求

| 函数 | 说明 |
|------|------|
| `llmapiReqNew(model)` / `llmapiReqFree(req)` | 请求生命周期 |
| `llmapiReqAddMsg(req, role, content)` | 添加消息 (role: system/user/assistant/tool) |
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
