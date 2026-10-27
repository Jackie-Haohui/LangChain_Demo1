import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage

load_dotenv()

# ================= 1. 建库和检索器 =================
documents = [
    Document(page_content="试用期员工可以申请办公设备，但单件价格不得超过 2000 元。",
             metadata={"source": "办公设备管理办法.md", "section": "试用期员工"}),
]
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3", api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", check_embedding_ctx_length=False
)
vector_store = InMemoryVectorStore.from_documents(documents=documents, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ================= 2. 使用官方自动工具,内置工具 =================
search_tool = create_retriever_tool(
    retriever=retriever,
    name="search_company_policies",
    description="检索公司制度。",
    response_format="content_and_artifact",  # 👈 核心在这里，它保留了原始 docs
)

# ================= 3. 创建 Agent 并运行 =================
model = ChatOpenAI(
    model="qwen-plus", api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
agent = create_agent(model=model, tools=[search_tool])

query = "试用期买显示器能超过2000吗？"
result = agent.invoke({"messages": [{"role": "user", "content": query}]})

# ================= 4. 演示：如何拿到文字回答和底层文档数据 =================
print("--- 🤖 最终回答（content） ---")
# 最终回答永远是最后一条消息
print(result["messages"][-1].content)

print("\n--- 📂 底层文档数据（artifact） ---")
# 遍历历史消息，找到工具执行的回执（ToolMessage）
for msg in result["messages"]:
    if isinstance(msg, ToolMessage):
        print(f"工具名称: {msg.name}")

        # 这就是你需要的原始数据！
        if msg.artifact:
            for i, doc in enumerate(msg.artifact):
                print(f"  文档 {i + 1}: {doc.page_content}")
                print(f"  元数据: {doc.metadata}")