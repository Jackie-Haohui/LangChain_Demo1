import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver # 导入内存检查点

# 1. 初始化千问
load_dotenv()
model = init_chat_model(
    model="qwen-plus", model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 创建 Agent，并绑定 Checkpointer（短期记忆）
agent = create_agent(
    model=model,
    tools=[],
    # 加上这行，Agent 就拥有了记忆能力
    checkpointer=InMemorySaver()
)

# 3. 配置会话 ID（这是记忆的钥匙）
config = {
    "configurable": {
        "thread_id": "conversation-1001", # 假设这是小林专属的聊天窗口
    }
}

print("========== 第一次对话 ==========")
# 4. 第一次提问：告诉 Agent 名字
print("用户：我叫小林，正在学习 LangChain。")
result1 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "我叫小林，正在学习 LangChain。"}
        ]
    },
    config=config, # 传入 config，告诉 Agent 存进哪个房间
)
print(f"AI: {result1['messages'][-1].content}")


print("\n========== 第二次对话 ==========")
# 5. 第二次提问：测试它记不记得
print("用户：我刚才说自己在学什么？")
result2 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "我刚才说自己在学什么？"}
        ]
    },
    config=config, # 再次传入相同的 config (thread_id)
)
print(f"AI: {result2['messages'][-1].content}")

# 附加：你可以打印 result2 的 messages 列表，看看它到底存了多少条记录
# for msg in result2["messages"]:
#     msg.pretty_print()