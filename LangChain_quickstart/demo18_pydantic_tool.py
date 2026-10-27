import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.chat_models import init_chat_model


# 1. 定义参数的 Schema
class OrderQuery(BaseModel):
    customer_id: str = Field(
        min_length=2,
        description="客户 ID，例如 C1001"
    )
    status: Literal["all", "pending", "shipped"] = Field(
        default="all",
        description="订单状态，all 表示不限制状态"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最多返回多少条订单"
    )


# 2. 定义工具，并绑定 Schema
@tool(args_schema=OrderQuery)
def search_orders(
        customer_id: str,
        status: str = "all",
        limit: int = 5
) -> list[dict]:
    """按客户 ID 查询订单，可按订单状态筛选。"""
    print(f"\n[底层执行] 正在数据库查询：客户={customer_id}, 状态={status}, 限制={limit}")

    # 模拟数据库里的数据
    all_orders = [
        {"order_id": "A1024", "customer_id": "C1001", "status": "shipped"},
        {"order_id": "A1025", "customer_id": "C1001", "status": "pending"},
        {"order_id": "A1026", "customer_id": "C1002", "status": "shipped"},
    ]

    # 过滤逻辑
    results = [o for o in all_orders if o["customer_id"] == customer_id]
    if status != "all":
        results = [o for o in results if o["status"] == status]

    return results[:limit]


# 3. 解剖工具，看看它的参数结构变成了什么样子
print("========== 工具参数结构 (args) ==========")
# 你会看到它生成了极其严谨的 JSON Schema
print(search_orders.args)

# 4. 测试本地执行与强校验
print("\n========== 测试 1：合法参数 ==========")
result = search_orders.invoke({"customer_id": "C1001", "status": "pending"})
print(f"查询结果: {result}")

print("\n========== 测试 2：非法参数 (触发 Pydantic 拦截) ==========")
try:
    # limit=50 超过了 le=20 的限制，会直接报错，根本不会执行函数内部代码
    search_orders.invoke({"customer_id": "C1001", "limit": 50})
except Exception as e:
    print(f"报错啦！拦截到非法数据：\n{e}")

# 5. 绑定大模型测试
print("\n========== 测试 3：绑定大模型 ==========")
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
model_with_tools = model.bind_tools([search_orders])

# 告诉模型：查 C1001 的已发货订单，最多查 10 条
response = model_with_tools.invoke("帮我查一下客户 C1001 的已发货订单，最多给我 10 条。")
print(f"大模型提取的工具调用: {response.tool_calls}")