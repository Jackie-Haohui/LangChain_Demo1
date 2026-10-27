import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

#只有相互独立的分支才适合

# 1. 初始化千问
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 定义两套 Prompt 模板
summary_prompt = ChatPromptTemplate.from_template("用一句话总结下面的内容：\n{text}")
keywords_prompt = ChatPromptTemplate.from_template("从下面内容中提取 3 个关键词，用逗号分隔：\n{text}")

# 3. 组装两个子链（注意：这里只到 prompt | model 为止，不急着转字符串，因为下面要统一处理）
# 这里为了演示方便，还是加上了 StrOutputParser，让每个分支直接输出文本
summary_chain = summary_prompt | model | StrOutputParser()
keywords_chain = keywords_prompt | model | StrOutputParser()

# 4. 并行组合
analysis = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
)

# 5. 测试文本
long_text = """
LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
它允许开发者将语言模型与其他来源的数据连接起来，并允许语言模型与其环境进行交互。
LangChain 的核心组件包括模型（Models）、提示词（Prompts）、索引（Indexes）、记忆（Memory）、链（Chains）和智能体（Agents）。
"""

# 6. 执行并计时
print("开始并行分析...")
start_time = time.time()

result = analysis.invoke({"text": long_text})

print(f"耗时: {time.time() - start_time:.2f} 秒\n")
print("=== 摘要 ===")
print(result["summary"])
print("\n=== 关键词 ===")
print(result["keywords"])