import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# 1. 定义工具
@tool
def get_weather(city: str) -> dict:
    """查询指定城市的当前天气。"""
    print(f"\n[底层执行] 正在运行真实的 Python 查天气函数：{city}...")
    weather_data = {
        "北京": {"condition": "晴", "temperature_c": 28},
        "上海": {"condition": "小雨", "temperature_c": 25},
    }
    # 如果没查到，返回默认值
    return weather_data.get(city, {"condition": "暂无数据", "temperature_c": None})

# 2. 初始化千问
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 3. 绑定工具
model_with_tools = model.bind_tools([get_weather])

# 4. 发起调用
print("--- 发起请求 ---")
response = model_with_tools.invoke("天津今天天气怎么样？")

# 5. 查看结果
print("\n--- 模型的原始回复对象 ---")
print(f"模型的文字内容 (content): {response.content}")
print(f"模型调用的工具 (tool_calls): {response.tool_calls}")

# --- 验证：模型并没有真正执行工具 ---
# 你会发现控制台并没有打印出 "[底层执行] 正在运行..."
# 因为现在的 response 只是一个“指令”，真正的执行需要你手动或者交给 Agent 去做。

# --- 接在你刚才的代码下面 ---

from langchain_core.messages import ToolMessage, HumanMessage

print("\n--- 开始执行闭环流程 ---")

# 1. 检查模型是否要求调用工具
if response.tool_calls:
    # 取出第一个工具调用的请求（如果模型同时调用了多个工具，这里需要循环）
    tool_call = response.tool_calls[0]

    # 2. 根据模型给的参数，在我们本地真正执行这个 Python 函数
    # 注意：这里需要用到你最开始定义的 get_weather 函数
    # 虽然它被 @tool 包装了，但它本身还是个 Runnable，可以直接 invoke
    tool_result = get_weather.invoke(tool_call["args"])

    print(f"[本地代码] 成功获取到真实数据：{tool_result}")

    # 3. 把执行结果包装成 ToolMessage（给模型看的回执单）
    # 必须带上 tool_call["id"]，让模型知道这是对应哪次请求的回复
    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"]
    )

    # 4. 将 AI 的决策和工具的执行结果一起发给模型
    # 这是一个对话历史：用户提问 -> AI决定用工具 -> 代码返回工具结果
    final_response = model_with_tools.invoke([
        HumanMessage("天津今天天气怎么样？"),
        response,  # 这是模型刚才返回的 AIMessage (包含 tool_calls)
        tool_message  # 这是工具执行完的结果
    ])

    # 5. 打印模型的最终回答
    print("\n--- 模型的最终回答 ---")
    print(final_response.content)