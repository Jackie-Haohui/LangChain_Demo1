import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# 读取 .env 中的环境变量
load_dotenv()

# 创建通义千问聊天模型对象（使用 OpenAI 兼容模式）
model = init_chat_model(
    model=os.getenv("DEFAULT_LLM_MODEL", "qwen3.8-27b"),  # 模型名称，可换为 qwen-max、qwen-turbo 等
    model_provider="openai",  # 指定提供者为 OpenAI 兼容协议
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 阿里云百炼 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云 DashScope 兼容端点
)

# 初始化消息列表，存入第一句话
messages = [
    HumanMessage("我叫小李，请记住我的名字。"),
]

print("--- 第一次提问 ---")
# 第一次调用模型
first_response = model.invoke(messages)
print(f"{first_response.content}")
print(f"{first_response.text}")
print(f"{first_response.json}")
print(f"{first_response.usage_metadata}")
print(f"{first_response.received_metadata}")

# 关键步骤：把 AI 的回复也加入到历史消息中
messages.append(first_response)

# 再次追加用户的第二句话
messages.append(HumanMessage("我叫什么名字？"))

print("\n--- 第二次提问 ---")
# 第二次调用模型，此时模型能看到完整的 messages 列表
second_response = model.invoke(messages)

print(f"AI回复: {second_response.content}")
