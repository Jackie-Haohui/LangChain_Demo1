import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.tools import tool

# 1. 模拟一个必定超时的工具
@tool
def search_orders(customer_id: str) -> list[dict]:
    """按客户 ID 查询最近的订单。"""
    print(f"\n[底层执行] 正在查询 {customer_id} 的订单... 啪！服务超时了！")
    raise TimeoutError("订单服务响应超时")

# 2. 定义防弹衣中间件
@wrap_tool_call
def handle_tool_error(request, handler):
    try:
        return handler(request)
    except TimeoutError:
        print("\n[中间件拦截] 捕获到超时异常，正在转化为友好的 ToolMessage...")
        return ToolMessage(
            content="订单服务暂时不可用，请稍后重试。",
            tool_call_id=request.tool_call["id"],
        )

# 3. 初始化模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus", model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 4. 组装 Agent
agent = create_agent(
    model=model,
    tools=[search_orders],
    middleware=[handle_tool_error],
)

# 5. 测试
print("--- 用户提问 ---")
print("帮我查一下客户 C1001 的订单。")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查一下客户 C1001 的订单。"}]}
)

print("\n--- 最终回答 ---")
print(result["messages"][-1].content)