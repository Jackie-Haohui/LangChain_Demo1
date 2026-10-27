import os
import random
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)


# ================= 1. 定义一个“极度不稳定”的工具 =================
@tool
def search_orders(customer_id: str) -> list[dict]:
    """按客户 ID 查询最近的订单。"""
    print(f"   🔎 [工具执行] 正在尝试查询客户 {customer_id} 的订单...")

    # 模拟 70% 的概率发生网络超时
    if random.random() < 0.7:
        raise TimeoutError("服务器连接超时！")

    return [{"order_id": "A1024", "status": "运输中"}]


# ================= 2. 初始化千问 =================
load_dotenv()
model = init_chat_model(
    model="qwen-plus", model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================= 3. 组装带三大护法的 Agent =================
agent = create_agent(
    model=model,
    tools=[search_orders],
    middleware=[
        # 1. 限制单次任务模型最多调 6 次，超了直接结束
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),

        # 2. 限制查订单工具最多跑 3 次
        ToolCallLimitMiddleware(tool_name="search_orders", run_limit=3),

        # 3. 只针对网络故障自动重试 2 次
        ToolRetryMiddleware(
            tools=["search_orders"],
            max_retries=2,
            retry_on=(TimeoutError, ConnectionError),
        ),
    ],
)

# ================= 4. 执行测试 =================
print("--- 用户提问 ---")
print("帮我查一下客户 C1001 的订单。\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查一下客户 C1001 的订单。"}]}
)

print("\n--- 最终结果 ---")
print(result["messages"][-1].content)