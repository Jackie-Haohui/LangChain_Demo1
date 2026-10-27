import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool


# 1. 定义工具
@tool
def get_weather(city: str) -> dict:
    """查询指定城市的当前天气。"""
    print(f"\n[底层执行] 正在查询 {city} 的真实天气...")
    return {"city": city, "condition": "晴", "temperature_c": 28}


# 2. 初始化千问
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 3. 绑定工具与建立字典映射
tools = [get_weather]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools) #把get_weather转换成大模型能看懂的 JSON Schema 格式

# 4. 初始化消息列表
messages = [
    {"role": "user", "content": "广州今天天气怎么样？"}
]

# 5. 第一次调用：让模型决定是否使用工具
print("--- 第一次请求：模型做决定 ---")
ai_message = model_with_tools.invoke(messages)
messages.append(ai_message)  # 把 AI 的决定加进历史记录

# 6. 执行工具并回传结果
print(f"模型决定调用工具: {ai_message.tool_calls}")
for tool_call in ai_message.tool_calls:
    selected_tool = tools_by_name[tool_call["name"]]

    # 传入整条调用记录，LangChain 会自动执行并生成完美的 ToolMessage
    tool_message = selected_tool.invoke(tool_call)

    print(f"[自动生成的回执单] {tool_message}")
    messages.append(tool_message)

# 7. 第二次调用：让模型根据工具结果组织回答
print("\n--- 第二次请求：模型做总结 ---")
final_message = model_with_tools.invoke(messages)

# 注意：新版 LangChain 用 .content
print(f"最终回答: {final_message.content}")