import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 1. 初始化模型、Prompt、Parser
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
prompt = ChatPromptTemplate.from_messages([("human", "请用一句话解释：{topic}")])
parser = StrOutputParser()

# 2. 定义普通函数（数据清洗+添加系统级提示词）
def preprocess_input(user_text: str) -> dict:
    # 去除乱七八糟的符号和空格
    cleaned = user_text.replace("?", "").replace("！", "").strip()
    print(f"[日志] 数据清洗完毕，准备发送给模型：{cleaned}")
    return {"topic": cleaned}

# 3. 把普通函数包装成 Runnable
cleaner = RunnableLambda(preprocess_input)

# 4. 组装超级链条
chain = cleaner | prompt | model | parser

# 5. 测试
messy_input = "   请问一下，什么是 闭包 ？？？   "
print(f"原始输入：'{messy_input}'")
print("--- 模型回答 ---")
print(chain.invoke(messy_input))