import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# ================= 1. 建库与 Retriever =================
documents = [
    Document(
        page_content="试用期员工可以申请办公设备，但单件价格不得超过 2000 元。申请应在采购系统中提交，并由直属主管审批。",
        metadata={"source": "办公设备管理办法.md", "section": "试用期员工"},
    ),
    Document(
        page_content="正式员工的标准设备额度为每两年 5000 元。超过额度的设备需要部门负责人追加审批。",
        metadata={"source": "办公设备管理办法.md", "section": "设备额度"},
    ),
    Document(
        page_content="办公设备由公司统一采购。员工自行购买后提交报销，只有在采购部门书面同意的情况下才会受理。",
        metadata={"source": "费用报销制度.md", "section": "设备报销"},
    ),
]

# 初始化千问 Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3", api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", check_embedding_ctx_length=False
)

# 一步建库
vector_store = InMemoryVectorStore.from_documents(documents=documents, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# ================= 2. 将 Retriever 封装为 Tool（核心） =================
@tool(response_format="content_and_artifact")
def search_company_policies(query: str) -> tuple[str, list[Document]]:
    """检索公司制度。涉及设备申请、采购和报销规则时使用。"""
    # 执行检索
    docs = retriever.invoke(query)

    if not docs:
        return "没有找到相关制度", []

    # 拼接给大模型看的纯文本
    content_parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        section = doc.metadata.get("section", "未标注章节")
        content_parts.append(
            f"[{index}] 来源：{source}；章节：{section}\n"
            f"内容：{doc.page_content}"
        )

    content = "\n\n".join(content_parts)

    # content 给模型阅读，docs (artifact) 留给应用层做展示
    return content, docs


# ================= 3. 创建 Agent 并调用 =================
model = ChatOpenAI(
    model="qwen-plus", api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 把刚才封装好的工具给 Agent
agent = create_agent(
    model=model,
    tools=[search_company_policies],
    system_prompt="你是公司行政助手。回答员工关于设备和报销的问题时，必须先调用工具检索公司制度。"
)

# ================= 4. 运行测试 =================
if __name__ == "__main__":
    query = "我是试用期员工，想买个 2500 块的显示器，能批吗？"
    print(f"👤 用户：{query}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    print("🤖 Agent 回答：")
    print(result["messages"][-1].content)

    # 如果你要在代码里获取原始引用文档，可以像下面这样从 tool_calls 里拿
    # （不过在 Agent 黑盒里，artifact 被吞了，只有手动调用工具时才能直接拿到 tuple）