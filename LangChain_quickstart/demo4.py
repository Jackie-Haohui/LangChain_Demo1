import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

# 1. 读取环境变量并初始化千问模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 定义提示词模板
prompt = PromptTemplate.from_template(
    "你是一名专业翻译。请把下面内容翻译成{target_language}，保持原意和语气，不要增加解释。\n待翻译内容：\n{text}"
)

# 3. 填充模板，生成完整的 PromptValue（注意这里还没有发请求）
formatted_prompt = prompt.invoke({
    "target_language": "英语",
    "text": "欢迎使用我们的产品"
})

# 4. 把填充好的提示词发给模型（真正联网的步骤在这里）
response = model.invoke(formatted_prompt)

# 5. 打印模型回答
print("--- 模型返回 ---")
print(response.content)