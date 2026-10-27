import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# ================= 1. 准备数据并建库（前置步骤） =================
docs = [
    Document(page_content="员工入职满六个月后，每年可享受 10 天带薪年假。年假应至少提前 3 个工作日申请。", metadata={"source": "休假制度"}),
    Document(page_content="单笔业务费用超过 5000 元时，需要部门负责人追加审批。", metadata={"source": "财务制度"}),
    Document(page_content="公司提供免费午餐，就餐地点在二楼食堂。", metadata={"source": "后勤制度"}),
    Document(page_content="病假需要提供三甲医院开具的证明，并在系统提交。", metadata={"source": "休假制度-病假"}),
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=10)
chunks = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False
)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=chunks)
print("✅ 向量库构建完成！\n")

# ================= 2. 核心：使用 Retriever 检索 =================
# 将 VectorStore 变成 Retriever，每次取最相似的 2 个片段
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

query = "我想请假去旅游，要提前几天走流程？"
print(f"--- 用户提问：{query} ---")

# 执行检索，拿到 Document 列表
retrieved_docs = retriever.invoke(query)

print("\n🎯 检索到的相关资料：")
for i, doc in enumerate(retrieved_docs):
    print(f"\n【片段 {i+1}】 来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content}")