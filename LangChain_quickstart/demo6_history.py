import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 初始化千问模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 构建带有历史记录占位符的模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名耐心的客服助手。"),
    # 这个占位符用来接收历史消息列表
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{question}"),
])

# 3. 模拟一段历史对话 (比如用户之前和客服的交流)
history = [
    HumanMessage("我的订单还没到。"),
    AIMessage("请提供订单号，我来帮你查询。"),
]

# 4. 将历史记录和当前问题填入模板
prompt_value = prompt.invoke({
    "history": history,
    "question": "订单号是 A1024。"
})

# === 以下为新增：将填充好的内容发给千问模型 ===
print("--- 发给模型的完整上下文 ---")
# 遍历打印看看里面到底装了什么
for message in prompt_value.to_messages():
    # 注意：新版 LangChain 获取文本内容用 .content
    print(f"[{type(message).__name__}]: {message.content}")

print("\n--- 模型最终回答 ---")
# 5. 调用模型
response = model.invoke(prompt_value)
print(f"[AIMessage]: {response.content}")