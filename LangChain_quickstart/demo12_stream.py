import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化千问
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 构建基础链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名技术编辑，解释要准确、简洁。"),
    ("human", "用不超过 150 个汉字解释：{topic}"),
])
parser = StrOutputParser()
chain = prompt | model | parser

print("========== 方式一：对整条链 stream（推荐） ==========\n")
# 这里直接拿到纯字符串片段
for chunk in chain.stream({"topic": "什么是事件循环"}):
    print(chunk, end="", flush=True)
print("\n\n")

print("========== 方式二：直接对 model stream 并累加 ==========\n")
full_message = None

# 这里循环拿到的是 AIMessageChunk 对象
for chunk in model.stream("请用一句话解释 Python 的垃圾回收机制",
    stream_options = {"include_usage": True}):

    # 边打印边累加
    print(chunk.content, end="", flush=True)

    # 累加逻辑（图三的代码，注意用 .content）
    if full_message is None:
        full_message = chunk
    else:
        full_message = full_message + chunk

print("\n\n--- 流式结束，这是累加后的完整结果 ---")
# 因为累加后依然是个 AIMessage 对象，你可以拿它的元数据
print(f"完整内容: {full_message.content}")
print(f"Token 消耗量: {full_message.usage_metadata}")