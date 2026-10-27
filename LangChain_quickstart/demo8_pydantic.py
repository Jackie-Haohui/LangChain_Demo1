import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal

# 1. 定义 Pydantic 数据模型 (图片里的代码)
class SupportTicket(BaseModel):
    """从用户反馈中提取的客服工单。"""
    category: Literal["account", "payment", "delivery", "other"] = Field(
        description="问题所属类别"
    )
    priority: int = Field(
        ge=1,
        le=5,
        description="紧急程度，1 最低，5 最高",
    )
    summary: str = Field(
        max_length=30,
        description="不超过 30 个汉字的问题摘要",
    )
    requires_human: bool = Field(
        description="是否需要人工客服立即介入"
    )

# 2. 初始化千问模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 3. 核心魔法：让模型输出结构化数据
# 这行代码告诉千问：你必须按照 SupportTicket 的格式给我返回数据
structured_model = model.with_structured_output(SupportTicket)

# 4. 模拟一条非常糟糕的用户反馈
user_complaint = "气死我了！我买的手机付了两次钱，到现在还没退给我。如果明天还不解决，我就去消费者协会投诉你们！赶紧叫个活人客服来跟我说话！"

# 5. 发起调用
print("正在分析用户反馈并提取工单...")
ticket = structured_model.invoke(user_complaint)

# 6. 打印结果
print("\n--- 提取结果 ---")
print(f"类型: {type(ticket)}") # 你会发现它不再是字符串，而是一个对象
print(f"问题类别: {ticket.category}")
print(f"紧急程度: {ticket.priority}")
print(f"问题摘要: {ticket.summary}")
print(f"需要人工: {ticket.requires_human}")