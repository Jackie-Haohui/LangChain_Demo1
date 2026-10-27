import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# 1. 定义嵌套的数据模型 (图中的代码)
class ActionItem(BaseModel):
    task: str = Field(description="需要完成的任务")
    owner: str | None = Field(
        default=None,
        description="明确指定的负责人，未指定时返回 null"
    )
    deadline: str | None = Field(
        default=None,
        description="原文中的截止日期，未提及时返回 null"
    )

class MeetingNotes(BaseModel):
    topic: str = Field(description="会议主题")
    decisions: list[str] = Field(description="会议中已经确认的决议列表")
    action_items: list[ActionItem] = Field(description="后续需要执行的任务清单")

# 2. 初始化千问模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 3. 绑定结构化输出
structured_model = model.with_structured_output(MeetingNotes)

# 4. 准备一段真实的会议废话
meeting_text = """
今天上午我们开了产品迭代会。大家一致决定，同意把深色模式在10月15日上线，
并且这个季度的营销预算增加20%。
接下来，小李负责去联系一下UI设计师，把深色模式的图标重新切图，周三前给到开发。
小王去写一份关于增加预算的申请报告，不用太急，下个月交就行。
另外，关于下个版本的底层架构重构，这次先不讨论，留着下次开会再说。
"""

# 5. 执行调用
print("正在提取会议纪要...")
notes = structured_model.invoke(meeting_text)

# 6. 打印结果
print("\n--- 会议主题 ---")
print(notes.topic)

print("\n--- 会议决议 ---")
for decision in notes.decisions:
    print(f"- {decision}")

print("\n--- 行动项 (嵌套对象) ---")
# 注意：notes.action_items 是个列表，里面的每个 item 都是一个 ActionItem 对象！
for item in notes.action_items:
    print(f"任务: {item.task}")
    print(f"负责人: {item.owner}")
    print(f"截止日期: {item.deadline}")
    print("-" * 20)