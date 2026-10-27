import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ================= 1. 前置准备：建库 =================
docs = [
    Document(page_content="员工入职满六个月后，每年可享受 10 天带薪年假。年假应至少提前 3 个工作日申请。",
             metadata={"source": "休假制度.md"}),
    Document(page_content="单笔业务费用超过 5000 元时，需要部门负责人追加审批。", metadata={"source": "财务制度.md"}),
]
chunks = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10).split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3", api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", check_embedding_ctx_length=False
)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# ================= 2. 辅助函数与 Prompt =================
def format_documents(docs: list[Document]) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        parts.append(f"[{index}] 来源：{doc.metadata.get('source', '未知')}\n内容：{doc.page_content}")
    return "\n\n".join(parts)


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，只能根据资料回答。\n资料：\n{context}"),
    ("human", "{question}")
])

model = ChatOpenAI(
    model="qwen-plus", api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ================= 3. 核心：LCEL 管道组装 =================
rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
)

# ================= 4. 测试运行 =================
if __name__ == "__main__":
    query = "我如果要休年假，最晚什么时候提交申请？"
    print(f"👤 用户提问：{query}\n")

    # 就这么简单，一行代码搞定检索+生成
    answer = rag_chain.invoke(query)

    print("🤖 AI 回答：")
    print(answer)