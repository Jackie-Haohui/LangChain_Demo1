import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. 定义 Pydantic 模型
class SupportTicket(BaseModel):
    category: str = Field(description="问题类别")
    summary: str = Field(description="问题摘要")

# 2. 初始化模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 3. 创建解析器并获取格式说明
parser = PydanticOutputParser(pydantic_object=SupportTicket)
format_instructions = parser.get_format_instructions()

# 4. 构建 Prompt，并强制插入格式说明
prompt = ChatPromptTemplate.from_messages([
    ("system", "分析用户反馈，并严格按照下面的格式返回。\n{format_instructions}"),
    ("human", "{feedback}"),
]).partial(format_instructions=format_instructions)

# 5. 组合链
chain = prompt | model | parser

# 6. 调用
try:
    ticket = chain.invoke({"feedback": "物流三天没有更新，希望尽快处理。"})
    print(f"提取成功！\n类型: {ticket.category}\n摘要: {ticket.summary}")
except Exception as e:
    # 如果模型废话太多，这里就会报错
    print(f"解析失败！错误信息:\n{e}")