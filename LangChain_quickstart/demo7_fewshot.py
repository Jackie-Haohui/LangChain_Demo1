import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 1. 初始化千问模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 构建包含“例题”的提示词模板
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "把用户反馈分类为 bug、feature 或 question，只返回分类结果，不要包含任何其他文字。",
    ),
    # 下面这三对就是 Few-shot 的“示例” (教你格式)
    ("human", "登录后页面一直空白"),
    ("ai", "bug"),
    ("human", "希望增加深色模式"),
    ("ai", "feature"),
    # 这是留出的真正填空题
    ("human", "{feedback}"),
])

# 3. 组合成链
chain = prompt | model

# 4. 填入真实数据并调用
print("正在让千问做分类...")
response = chain.invoke({
    "feedback": "导出的文件保存在哪里？"
})

# 5. 打印结果 (注意用 .content)
print(f"千问的分类结果: {response.content}")