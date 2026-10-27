import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

# 1. 模拟数据库
ORDERS = {
    "A1024": {
        "status": "运输中",
        "destination": "上海",
        "shipped_at": "2026-07-15",
    }
}
DELIVERY_ESTIMATES = {
    "上海": "预计 1 天内送达",
    "北京": "预计 2 天内送达",
}

# 2. 定义工具 1：查订单
@tool
def get_order(order_id: str) -> dict:
    """按订单号查询订单状态、收货城市和发货日期。"""
    order = ORDERS.get(order_id)
    if order is None:
        return {"ok": False, "message": "未找到该订单"}
    return {"ok": True, "order_id": order_id, **order}

# 3. 定义工具 2：预估送达时间
@tool
def estimate_delivery(destination: str) -> dict:
    """根据收货城市查询预计送达时间。"""
    estimate = DELIVERY_ESTIMATES.get(destination)
    if estimate is None:
        return {"ok": False, "message": "暂无该城市的配送时效"}
    return {"ok": True, "destination": destination, "estimate": estimate}

# 4. 初始化千问模型
load_dotenv()
# 注意：create_agent 通常需要传入模型对象。如果你用的 langchain 版本较新，可以直接传模型字符串配合 provider，但最稳妥的是直接传初始化好的模型对象或模型名。
# 这里按照图中的简化方式，如果用 create_agent，通常需要底层模型支持。
# 为了适配你的千问，我们手动初始化模型对象。
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 5. 创建 Agent
# 注意：如果你用的是最新版 LangChain，API 可能是 create_tool_calling_agent 等，图里使用的是较新的高层封装 create_agent
agent = create_agent(
    model=model, # 换成千问模型对象
    tools=[get_order, estimate_delivery],
    system_prompt=(
        "你是订单服务助手。"
        "涉及订单状态和配送时效时必须使用工具查询，不得编造。"
        "工具返回失败时如实说明原因。"
        "最终使用简洁的中文回答。"
    ),
    #debug=True #可看到过程
)

# 6. 运行 Agent
print("--- 用户提问 ---")
print("订单 A1024 到哪里了，大概什么时候能送达？")

# 注意：Agent 的输入必须是字典，且包含 messages 列表
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "订单 A1024 到哪里了，大概什么时候能送达？",
            }
        ]
    },
    #限制最大运行次数
    config={"recursion_limit":10},
)

agent_input = {
    "messages": [
        {
            "role": "user",
            "content": "订单 A1024 到哪里了，大概什么时候送到？"
        }
    ]
}

# # 7. 打印最终结果
# print("\n--- Agent 最终回答 ---")
# # 注意：图里写的是 .text，新版请用 .content
# print(result["messages"][-1].content)


print("\n========== 完整的历史记录 ==========")
for message in result["messages"]:
    message.pretty_print()

print("\n========== 只提取最终一句话 ==========")
# 注意：这里我用的是 .content，如果你报错就换成 .text
final_answer = result["messages"][-1].content
print(final_answer)

# for state in agent.stream(agent_input, stream_mode="values"):
#     latest_message = state["messages"][-1]
#     latest_message.pretty_print()