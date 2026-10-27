import os
import time
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

prompt = ChatPromptTemplate.from_template("请用一句话解释什么是{topic}")
parser = StrOutputParser()
chain = prompt | model | parser

# 2. 准备多组互不依赖的输入数据
inputs = [
    {"topic": "装饰器"},
    {"topic": "上下文管理器"},
    {"topic": "协程"},
    {"topic": "垃圾回收"},
]

print("========== 方式一：传统 for 循环（串行，慢） ==========")
start_time = time.time()
for item in inputs:
    result = chain.invoke(item)
    print(f"- {item['topic']}: {result[:20]}...")
print(f"传统耗时: {time.time() - start_time:.2f} 秒\n")


print("========== 方式二：chain.batch（并发，有序） ==========")
start_time = time.time()
# 传入列表，并限制最大并发数为 3
results = chain.batch(inputs, config={"max_concurrency": 3})
for item, result in zip(inputs, results):
    # 你会发现结果和输入完美对应，按顺序返回
    print(f"- {item['topic']}: {result[:20]}...")
print(f"batch耗时: {time.time() - start_time:.2f} 秒\n")


print("========== 方式三：chain.batch_as_completed（谁先好谁上） ==========")
start_time = time.time()
# 循环接收结果。注意这里是 for 循环直接拿
for index, result in chain.batch_as_completed(inputs, config={"max_concurrency": 3}):
    # index 是原始输入列表里的索引，用来对齐问题
    topic = inputs[index]['topic']
    print(f"[单号 {index} 完成] {topic}: {result[:20]}...")
print(f"as_completed耗时: {time.time() - start_time:.2f} 秒")