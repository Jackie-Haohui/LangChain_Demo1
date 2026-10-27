import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model, before_model
from langgraph.runtime import Runtime
from langchain.tools import tool

# 1. 定义中间件
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:  #返回None代表此函数不做任何修改
    # 每次调用模型前，打印当前系统里存了多少条消息
    print(f"🟢 [Middleware] 准备调用模型，当前消息数: {len(state['messages'])}")

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    latest_message = state["messages"][-1]
    # 看看模型这次返回的是纯文本还是工具调用
    print(f"🔴 [Middleware] 模型返回类型: {type(latest_message).__name__}")

# 2. 随便定义一个工具，为了触发多轮循环
@tool
def get_time() -> str:
    """获取当前时间"""
    return "2026-10-26 12:00:00"

# 3. 初始化模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus", model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 4. 创建 Agent 并挂载中间件
agent = create_agent(
    model=model,
    tools=[get_time],
    middleware=[log_before_model, log_after_model], # 👈 挂载
)

# 5. 执行
print("--- 用户提问 ---")
print("现在几点了？")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "现在几点了？"}]}
)

print("\n--- 最终回答 ---")
print(result["messages"][-1].content)