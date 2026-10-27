import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)

load_dotenv()

# 1. 初始化弱模型（按你的要求）
fast_model = init_chat_model(
    model="qwen3.8-flash",  # 如果报错找不到，换成 "qwen-turbo" 或 "qwen-flash"
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 初始化强模型（默认使用）
strong_model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 3. 定义中间件（完全复刻图片逻辑）
@wrap_model_call
def choose_model(
        request: ModelRequest,
        handler,
) -> ModelResponse:
    # 获取当前上下文里的消息数量
    message_count = len(request.state["messages"])

    # 图片逻辑：如果消息数量大于 10 条，用强模型；否则用弱模型
    selected_model = strong_model if message_count > 10 else fast_model

    print(f"   🤖 [中间件] 当前消息数: {message_count}，调度模型: {selected_model.model_name}")

    # override 只修改本次模型请求
    return handler(request.override(model=selected_model))


# 4. 创建 Agent
# 注意：这里默认先传 strong_model（因为你要求默认用 qwen-plus）
agent = create_agent(
    model=strong_model,
    tools=[],
    middleware=[choose_model],
)

# 5. 测试运行

# --- 场景 A：只有 1 条消息（应触发 fast_model）---
print("\n--- 场景 A：刚开始对话 ---")
res1 = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print(f"AI回答: {res1['messages'][-1].content}")

# --- 场景 B：制造 11 条消息（应触发 strong_model）---
print("\n--- 场景 B：多轮长对话 ---")
# 为了模拟长对话，我们构造一个包含 11 条历史的列表
fake_history = []
for i in range(5):
    fake_history.append({"role": "user", "content": f"问题 {i}"})
    fake_history.append({"role": "assistant", "content": f"回答 {i}"})
fake_history.append({"role": "user", "content": "这是第 11 条消息，请仔细思考后回答。"})  # 此时一共 11 条

res2 = agent.invoke({"messages": fake_history})
print(f"AI回答: {res2['messages'][-1].content}")