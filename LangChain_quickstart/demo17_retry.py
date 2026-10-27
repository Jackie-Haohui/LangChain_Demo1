import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 1. 初始化千问
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 模拟一个不稳定的外部系统（前两次必报错，第三次才成功）
attempt_count = 0


def unstable_node(inputs):
    global attempt_count
    attempt_count += 1
    print(f"[系统日志] 尝试第 {attempt_count} 次调用...")

    if attempt_count < 3:
        print("[系统报错] 遭遇了网络抖动或限流！")
        raise Exception("502 Bad Gateway")  # 模拟报错

    print("[系统恢复] 第三次尝试成功！")
    return inputs  # 成功，把数据传给下一个节点


unstable = RunnableLambda(unstable_node)

prompt = ChatPromptTemplate.from_template("请用一句话解释：{topic}")
parser = StrOutputParser()

# 3. 组装基础链（假设这个不稳定节点是链条的一部分）
base_chain = unstable | prompt | model | parser

# 4. 给整条链加上重试机制！
reliable_chain = base_chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

# 5. 执行
print("--- 开始执行 ---")
result = reliable_chain.invoke({"topic": "什么是幂等性"})
print("\n--- 最终结果 ---")
print(result)