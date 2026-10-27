from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 1. 构造原始 Document =================
print("--- 步骤 1：构建原始 Document ---")
documents = [
    Document(
        page_content=(
            "员工入职满六个月后，每年可享受 10 天带薪年假。"
            "年假应至少提前 3 个工作日，在 OA 系统提交申请，"
            "并经直属主管批准后生效。"
            "此外，对于在公司工作满三年的员工，每满一年增加 1 天年假，最多不超过 15 天。"
        ),
        metadata={"source": "员工手册.md", "section": "休假制度"},
    ),
    Document(
        page_content=(
            "业务费用应在发生后 30 天内提交报销。"
            "单笔金额超过 5000 元时，需要部门负责人追加审批。"
            "差旅费的报销需要附上发票原件及出差审批单。"
        ),
        metadata={"source": "财务制度.md", "section": "费用报销"},
    ),
]

print(f"原始文档数量: {len(documents)}")
for doc in documents:
    print(f"来源: {doc.metadata['source']}, 长度: {len(doc.page_content)} 字符")


# ================= 2. 文本切分 =================
print("\n--- 步骤 2：进行文本切分 ---")
# 故意把 chunk_size 设小一点，方便观察切分效果
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,      # 每段最多 40 个字符
    chunk_overlap=10,   # 相邻片段重叠 10 个字符
    add_start_index=True, # 记录在原文中的起始位置
)

chunks = text_splitter.split_documents(documents)

print(f"切分后的文本块数量: {len(chunks)}\n")

# ================= 3. 查看切分结果 =================
for i, chunk in enumerate(chunks):
    print(f"【片段 {i+1}】")
    print(f"  内容: {chunk.page_content}")
    print(f"  元数据: {chunk.metadata}")
    print("-" * 30)