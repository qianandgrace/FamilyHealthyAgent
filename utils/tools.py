import os
import asyncio
import warnings

from langchain_chroma import Chroma
from langchain_core.tools import tool

from utils.llm import get_llm
from configs.config_loader import config

_, llm_embedding = get_llm("qwen")
vectorstore = Chroma(
                    persist_directory="chromaDB",
                    collection_name= "health_records" ,
                    embedding_function=llm_embedding)

######### 自定义工具 ##########

PERSIST_DIR = config["chromadb"]["directory"]
# 三个collection
knowledge_store = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name="knowledge",
    embedding_function=llm_embedding,
)
redflag_store = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name="redflag",
    embedding_function=llm_embedding,
)
triage_store = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name="triage",
    embedding_function=llm_embedding,
)

def _format_docs(docs):
    # 统一格式，便于 LLM 阅读
    return "\n\n".join(
        [f"Source: {d.metadata.get('category')}\nMetadata: {d.metadata}\n" for d in docs]
    )

@tool("retrieve_knowledge", description="医学知识库检索：疾病/症状/治疗/就医建议等。")
def retrieve_knowledge(query: str) -> str:
    retriever = knowledge_store.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    return _format_docs(docs)


@tool("retrieve_redflag", description="红旗信号检索：用于判断是否存在急危重症信号以及建议动作。")
def retrieve_redflag(query: str) -> str:
    retriever = redflag_store.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    return _format_docs(docs)


@tool("retrieve_triage", description="分诊规则检索：根据症状/人群匹配 home/clinic/emergency 规则。")
def retrieve_triage(query: str) -> str:
    retriever = triage_store.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    return _format_docs(docs)


def get_tools():
    return [retrieve_redflag, retrieve_triage, retrieve_knowledge]