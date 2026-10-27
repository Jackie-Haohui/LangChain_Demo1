import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


@dataclass
class RequestContext:
    user_id: str


# 👇 核心防呆逻辑：把各种可能的叫法，统一映射为固定的 Key
PREF_ALIASES = {
    "answer_style": "answer_style",
    "回答风格": "answer_style",
    "风格": "answer_style",
    "偏好": "answer_style"
}


def normalize_name(name: str) -> str:
    """将大模型可能生成的各种偏好名，统一标准化"""
    return PREF_ALIASES.get(name, name)


@tool
def save_preference(
        preference_name: str,
        preference_value: str,
        runtime: ToolRuntime[RequestContext],
) -> str:
    """保存当前用户明确表达的长期偏好。"""
    if runtime.store is None:
        return "长期存储不可用"

    # 强制归一化参数名
    safe_name = normalize_name(preference_name)
    print(f"\n[💾 保存] 原始名称: {preference_name} -> 归一化后: {safe_name}")

    namespace = ("user_preferences", runtime.context.user_id)
    runtime.store.put(namespace, safe_name, {"value": preference_value})
    return "偏好已保存"


@tool
def get_preference(
        preference_name: str,
        runtime: ToolRuntime[RequestContext],
) -> str:
    """读取当前用户已经保存的长期偏好。"""
    if runtime.store is None:
        return "长期存储不可用"

    # 强制归一化参数名
    safe_name = normalize_name(preference_name)
    print(f"\n[🔍 查询] 原始名称: {preference_name} -> 归一化后: {safe_name}")

    namespace = ("user_preferences", runtime.context.user_id)
    item = runtime.store.get(namespace, safe_name)
    return item.value["value"] if item else "尚未保存该偏好"


load_dotenv()
model = init_chat_model(
    model="qwen-plus", model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

store = InMemoryStore()
agent = create_agent(
    model=model,
    tools=[save_preference, get_preference],
    context_schema=RequestContext,
    store=store,
    checkpointer=InMemorySaver(),
    # 这里依然保留强力 Prompt
    system_prompt=(
        "你是一个严格遵守指令的助手，你没有原生记忆，所有关于用户偏好的信息都存放在外部数据库中。\n"
        "1. 只要用户表达要保存偏好（如‘请记住’），你【必须】调用 `save_preference`。\n"
        "2. 只要用户询问任何关于偏好、喜好的问题（如‘我喜欢什么’），你【必须】先调用 `get_preference` 去查询数据库！\n"
        "【绝对禁止】在没有调用 `get_preference` 工具的情况下，直接回答‘不知道’或‘没有记录’。"
    ),
    debug=False,  # 关闭 debug，看纯净的日志
)

context = RequestContext(user_id="user_1001")

print("========== 会话 A：用户提出了偏好 ==========")
agent.invoke(
    {"messages": [{"role": "user", "content": "请记住我喜欢简洁的回答。"}]},
    context=context,
    config={"configurable": {"thread_id": "thread-a"}},
)
print("会话 A 执行完毕。")

print("\n========== 会话 B：换了一个新会话线程 ==========")
res_b = agent.invoke(
    {"messages": [{"role": "user", "content": "我喜欢什么样的回答？"}]},
    context=context,
    config={"configurable": {"thread_id": "thread-b"}},
)
print(f"\nAI最终回答: {res_b['messages'][-1].content}")