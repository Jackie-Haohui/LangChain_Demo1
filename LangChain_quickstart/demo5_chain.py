import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

# 1. 初始化模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 定义模板
prompt = PromptTemplate.from_template(""
         "你是一个{domain}专家。请回答：{question}"
                                      "")

# 3. 使用管道符组合：把 prompt 的输出直接交给 model
chain = prompt | model

# 4. 执行整条链。只需要传入字典，它自己会去填空 -> 发给模型 -> 拿回结果
response = chain.invoke({
    "domain": "数据库",
    "question": "索引为什么能提高查询速度？"
})

print("--- 管道符调用结果 ---")
print(response.content)