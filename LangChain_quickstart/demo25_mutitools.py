import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import AIMessage, ToolMessage

# ================= 1. 模拟数据库 =================
PRODUCTS = {
    "P1001": {"name": "机械键盘", "price_yuan": 499.0, "stock": 8}
}
MEMBER_DISCOUNTS = {
    "C1001": 0.9,
    "C1002": 0.95,
}

# ================= 2. 定义三个工具 =================
@tool
def get_product(product_id: str) -> dict:
    """按商品 ID 查询商品名称、原价和当前库存。"""
    print(f"\n[⚙️ 工具执行] 正在查询商品: {product_id}")
    product = PRODUCTS.get(product_id)
    if product is None:
        return {"ok": False, "message": "未找到该商品"}
    return {"ok": True, "product_id": product_id, **product}

@tool
def get_member_discount(customer_id: str) -> dict:
    """按客户 ID 查询当前有效的会员折扣率。"""
    print(f"\n[⚙️ 工具执行] 正在查询会员: {customer_id}")
    discount_rate = MEMBER_DISCOUNTS.get(customer_id)
    if discount_rate is None:
        return {"ok": False, "message": "未找到会员信息"}
    return {"ok": True, "customer_id": customer_id, "discount_rate": discount_rate}

@tool
def calculate_final_price(price_yuan: float, discount_rate: float) -> dict:
    """根据商品原价和折扣率计算最终价格。"""
    print(f"\n[⚙️ 工具执行] 正在算账: {price_yuan} * {discount_rate}")
    final_price = round(price_yuan * discount_rate, 2)
    return {
        "price_yuan": price_yuan,
        "discount_rate": discount_rate,
        "final_price_yuan": final_price,
    }

# ================= 3. 初始化千问 =================
load_dotenv()
model = init_chat_model(
    model="qwen-plus", model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================= 4. 组装 Agent =================
agent = create_agent(
    model=model,
    tools=[get_product, get_member_discount, calculate_final_price],
    system_prompt=(
        "你是商品报价助手。\n"
        "报价前必须查询商品和会员折扣。\n"
        "金额计算必须使用 calculate_final_price。\n"
        "没有库存时不要报价，并如实说明。"
    ),
)

# ================= 5. 执行测试 =================
print("--- 用户提问 ---")
user_query = "客户 C1001 购买商品 P1001，到手价是多少？"
print(user_query)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": user_query}
        ]
    }
)

print("\n--- 最终回答 ---")
# 👈 修正了原图的 .text 报错
print(result["messages"][-1].content)

print("\n" + "=" * 30)
print("🕵️ 正在检查 Agent 的底层调用路径...")
print("=" * 30)

for message in result["messages"]:
    if isinstance(message, AIMessage) and message.tool_calls:
        # 发现了 AI 的工具调用决定
        for tool_call in message.tool_calls:
            print(f"👉 [AI 决定调用工具] : {tool_call['name']}")
            print(f"   [传了什么参数] : {tool_call['args']}")

    if isinstance(message, ToolMessage):
        # 发现了工具执行的结果
        print(f"✅ [工具执行完毕] : {message.name}")
        print(f"   [返回的结果] : {message.content[:50]}...")  # 截取前50个字防刷屏
        print("-" * 20)