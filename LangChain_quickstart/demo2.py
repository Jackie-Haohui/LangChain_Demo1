import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# 1. 读取环境变量
load_dotenv()

# 2. 封装构建模型的函数
def build_model(model_name: str):
    return init_chat_model(
        model=model_name,                   # 模型名称
        model_provider="openai",            # 千问兼容模式底层使用 openai 协议
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 传入阿里云百炼的 Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=30,                         # 新增：设置超时时间为 30 秒
        max_retries=3                       # 新增：设置失败自动重试 3 次
    )

# 3. 使用函数初始化模型 (换成你的千问模型)
model = build_model("qwen-plus")

# 4. 发起调用，这次加入了系统提示词
response = model.invoke(
    [
        # SystemMessage 用于设定 AI 的人设和角色
        SystemMessage("你是一名简洁的技术助手。"),
        # HumanMessage 是用户的具体提问
        HumanMessage("解释依赖注入的作用。")
    ]
)

# 5. 打印结果 (注意：新版 LangChain 推荐用 .content)
print(response.content)