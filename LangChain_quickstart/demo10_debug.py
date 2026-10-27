import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# 1. 定义一个限制极严格的数据模型
class StrictTicket(BaseModel):
    # 摘要最多 5 个字，大模型肯定会超标，以此来触发报错
    summary: str = Field(max_length=5, description="不超过 5 个字的极简摘要")

# 2. 初始化模型
load_dotenv()
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 3. 开启 include_raw=True
debug_model = model.with_structured_output(
    StrictTicket,
    include_raw=True,
)

# 4. 发起调用
result = debug_model.invoke("我的手机掉进水里了，现在开不了机，急需退款或者维修！")

# 5. 查看结果
print("=== 解析结果 (parsed) ===")
print(result["parsed"]) # 会返回 None，因为解析失败了

print("\n=== 错误信息 (parsing_error) ===")
# 这里会提示 summary 长度超过 5 个字
print(result["parsing_error"])

print("\n=== 原始响应 (raw) ===")
# 注意：raw 是 AIMessage 对象，要看里面的内容需要用 .content
raw_message = result["raw"]
print(raw_message.content)