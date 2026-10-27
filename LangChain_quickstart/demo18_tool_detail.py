import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool


# ================= 1. 定义并“变身”工具 =================
@tool
def estimate_shipping_fee(
        weight_kg: float,
        express: bool = False
) -> dict:
    """根据包裹重量计算配送费，可选择普通配送或加急配送。"""
    base_fee = 8 + weight_kg * 2
    if express:
        base_fee += 12

    return {
        "fee_yuan": round(base_fee, 2),
        "express": express,
    }


# ================= 2. 验证工具属性 =================
print("========== 工具属性解剖 ==========")
print(f"1. 工具名称 (name): {estimate_shipping_fee.name}")
print(f"2. 工具描述 (description): {estimate_shipping_fee.description}")
print(f"3. 参数结构 (args): {estimate_shipping_fee.args}")

# ================= 3. 手动调用测试 =================
print("\n========== 手动调用 (使用字典传参) ==========")
# 注意：给 Tool 对象传参，必须是字典格式
result = estimate_shipping_fee.invoke({
    "weight_kg": 2.5,
    "express": True
})
print(f"计算结果: {result}")

# ================= 4. 结合大模型调用测试 =================
print("\n========== 绑定大模型测试 ==========")
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

model_with_tools = model.bind_tools([estimate_shipping_fee])

# 问一个问题，看看大模型是否认出了这个工具
response = model_with_tools.invoke("我有个2.5公斤的包裹，想加急，运费多少？")
print(f"模型提取的工具调用: {response.tool_calls}")