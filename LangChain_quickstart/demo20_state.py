import os
from dotenv import load_dotenv
from typing import Annotated
from langchain.agents import AgentState, create_agent
from langchain.tools import InjectedState,tool
from langchain.chat_models import init_chat_model


#给 Agent 的执行引擎（后台代码）扩充一个“全局变量字典”，方便你在写业务代码（工具）时随时读取。

# ================= 1. 定义 State 的格式 (state_schema) =================
# 继承默认的 AgentState (里面原本就有 messages)
# 然后我们强行塞进去两个业务字段
class BankState(AgentState):
    is_vip: bool  # 是否是 VIP
    user_id: str  # 用户的真实系统 ID


# ================= 2. 定义工具，并读取 State =================
@tool
def withdraw_money(amount: float, state: Annotated[dict, InjectedState]) -> str:
    """取款工具。"""
    # 【核心机密在这里】：InjectedState 允许你的工具直接偷看后台的 State 字典！
    # 它根本不需要大模型把 user_id 传进来，而是系统直接注入进去的
    real_user_id = state.get("user_id")
    is_vip = state.get("is_vip")

    print(f"\n[后台拦截] 收到取款请求。系统识别到用户ID: {real_user_id}, VIP状态: {is_vip}")

    if is_vip:
        return f"VIP客户专属通道已打开，为您快速取出 {amount} 元。"
    else:
        return f"普通客户，请排队，已为您取出 {amount} 元。"


# ================= 3. 创建 Agent =================
load_dotenv()
model = init_chat_model(
    model="qwen-plus", model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

agent = create_agent(
    model=model,
    tools=[withdraw_money],
    # 【关键点】：告诉 Agent，以后运行的时候，别只带 messages，要把我的 BankState 当作流程模板
    state_schema=BankState
)

# ================= 4. 发起调用 =================
print("--- 用户说话 ---")
print("“帮我取 100 块钱。” (注意：用户没说自己是VIP，也没报ID)")

result = agent.invoke(
    {
        # 这是用户说的话
        "messages": [{"role": "user", "content": "帮我取 100 块钱。"}],

        # 这是后台系统（你的代码）强塞进去的真实业务数据
        "is_vip": True,  # 系统查出他是VIP
        "user_id": "U_9527"  # 系统查出的真实ID
    }
)

print("\n--- 最终回答 ---")
print(result["messages"][-1].content)