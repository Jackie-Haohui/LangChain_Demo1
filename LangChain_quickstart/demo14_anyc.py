import os
import time
import asyncio
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

prompt = ChatPromptTemplate.from_template("请用100字解释{topic}")
parser = StrOutputParser()
chain = prompt | model | parser

inputs = [
    {"topic": "什么是闭包"},
    {"topic": "什么是装饰器"},
    {"topic": "什么是协程"},
]

# ================== 方式一：同步串行（慢慢等） ==================
print("========== 同步串行 ==========")
start = time.time()
for item in inputs:
    chain.invoke(item)
print(f"同步耗时: {time.time() - start:.2f} 秒")

# ================== 方式二：异步并发（同时干） ==================
# 定义一个异步函数来处理所有请求
async def run_async():
    print("\n========== 异步并发 ==========")
    start = time.time()
    # 用 asyncio.gather 把多个 ainvoke 任务打包，让它们同时跑！
    tasks = [chain.ainvoke(item) for item in inputs]
    await asyncio.gather(*tasks)
    print(f"异步耗时: {time.time() - start:.2f} 秒")

# 启动异步环境
asyncio.run(run_async())