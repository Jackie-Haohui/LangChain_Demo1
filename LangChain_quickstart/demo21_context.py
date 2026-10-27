import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain.chat_models import init_chat_model


# ================= 1. 定义 Context =================
@dataclass
class RequestContext:
    user_id: str
    tenant_id: str


# ================= 2. 模拟数据库 =================
# 键是 (租户ID, 用户ID, 订单号)，这样彻底隔离不同用户的数据
ORDERS = {
    ("tenant_a", "user_1001", "A1024"): {
        "status": "运输中",
        "amount_yuan": 299,
    },
    ("tenant_a", "user_1002", "A1025"): {
        "status": "已签收",
        "amount_yuan": 150,
    }
}


# ================= 3. 定义工具 =================
@tool
def get_my_order(
        order_id: str,
        runtime: ToolRuntime[RequestContext],  # 注入运行时上下文
) -> dict:
    """查询当前登录用户自己的订单。"""
    context = runtime.context

    # 核心安全逻辑：强制使用 context 里的 user_id 和 tenant_id 去查库
    order = ORDERS.get((context.tenant_id, context.user_id, order_id))

    if order is None:
        return {"ok": False, "message": "未找到该订单，或您无权查看"}

    return {"ok": True, "order_id": order_id, **order}


# ================= 4. 初始化千问 =================
load_dotenv()
model = init_chat_model(
    model="qwen-plus", model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================= 5. 创建 Agent =================
agent = create_agent(
    model=model,
    tools=[get_my_order],
    context_schema=RequestContext,  # 绑定 Schema
)

# ================= 6. 运行测试 =================
print("=== 场景 1：user_1001 查询自己的订单 A1024 ===")
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "帮我查一下订单 A1024"}]},
    context = RequestContext(user_id="user_1001", tenant_id="tenant_a")
)
print(result1["messages"][-1].content)

print("\n=== 场景 2：user_1001 恶意查询 user_1002 的订单 A1025 ===")
# 模型依然会提取 order_id="A1025"，但是 context 依然是 user_1001
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "帮我查一下订单 A1025"}]},
     context=RequestContext(user_id="user_1001", tenant_id="tenant_a")
)
print(result2["messages"][-1].content)