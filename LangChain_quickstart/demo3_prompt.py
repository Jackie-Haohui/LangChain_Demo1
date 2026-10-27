from langchain_core.prompts import PromptTemplate

# 1. 定义模板，使用 {} 挖好坑
prompt = PromptTemplate.from_template(
    """
你是一名专业翻译。
请把下面内容翻译成{target_language}，保持原意和语气，不要增加解释。
待翻译内容：
{text}
""".strip()
)

# 2. 填入内容（调用 invoke 方法）
result = prompt.invoke({
    "target_language": "英语",
    "text": "欢迎使用我们的产品"
})

# 3. 获取生成的普通字符串
# 注意：图片里写的是 result.text，但在新版 LangChain 中，更推荐用 .to_string()
print("--- 填充后的完整提示词 ---")
print(result.to_string())

# 图片底部截断的部分：也可以直接用 format() 方法获取普通字符串，效果一样
print("\n--- 使用 format 方法 ---")
print(prompt.format(target_language="日语", text="欢迎使用我们的产品"))