# 功能说明：将PDF文件进行向量计算并持久化存储到向量数据库（chroma）
import os
import sys
import uuid
import logging
from tqdm import tqdm
from pathlib import Path

import chromadb

# 将项目根目录加入 sys.path，保证包内引用可用（以便从项目根运行脚本）
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir)) # 调试断点，检查路径设置是否正确
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.data_loader import *
from utils.llm import get_llm
from configs.config_loader import config

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从配置文件中读取全局变量
LLMTYPE = config['llm']['type']
CHROMADB_DIRECTORY = config['chromadb']['directory']

def chroma_safe_value(v):
    """把值转换为 Chroma 可接受的 metadata value: str/int/float/bool/None"""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Path):
        return str(v)
    # 兜底：dict/list/其他类型全部转字符串（更稳，不会炸）
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)


def sanitize_metadata(md: dict) -> dict:
    return {k: chroma_safe_value(v) for k, v in (md or {}).items()}

llm, llm_embedding = get_llm(LLMTYPE)   
# get_embeddings方法计算向量
def get_embeddings(texts):
    data = llm_embedding.embed_documents(texts)
    return data

# 对文本按批次进行向量计算
def generate_vectors(data, max_batch_size=25):
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        # 调用向量生成get_embeddings方法  根据调用的API不同进行选择
        response = get_embeddings(batch)
        results.extend(response)
    return results


# 封装向量数据库chromadb类，提供两种方法
class MyVectorDBConnector:
    def __init__(self, collection_name: str, embedding_fn):
        client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
        self.collection = client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str] | None = None
    ):
        if not texts:
            return
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(
            embeddings=self.embedding_fn(texts),
            documents=texts,
            metadatas=metadatas, # type: ignore
            ids=ids,
        )

    def reset(self):
        # 清空collection（重新灌库用）
        # chromadb python 原生 collection 没有 clear，通常用 delete where={}
        try:
            self.collection.delete(where={})
        except Exception:
            # 有些版本需要 where_document
            try:
                self.collection.delete(where_document={})
            except Exception as e:
                logger.warning(f"Reset collection failed: {e}")


# ---------- Build all ----------
def build_transfer_data_vectorstores(reset: bool = False):
    data_dir = Path(config['text']['input_md'])
    if not data_dir.exists():
        raise FileNotFoundError(f"{config['text']['input_md']} not found")
    knowledge_db = MyVectorDBConnector("knowledge", generate_vectors)
    redflag_db   = MyVectorDBConnector("redflag", generate_vectors)
    triage_db    = MyVectorDBConnector("triage", generate_vectors)
    if reset:
        logger.info("Resetting collections...")
        knowledge_db.reset()
        redflag_db.reset()
        triage_db.reset()
    # 1) knowledge: md files
    md_files = list(data_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} md files")
    for p in tqdm(md_files, desc="Processing MD files"):
        texts, metas = load_md_as_chunks(p)
        knowledge_db.add_texts(texts, metas)
    logger.info("MD files ingested into knowledge")
    # 2) knowledge: medical_data.jsonl
    # 为上一级目录
    base_data_dir = Path(config['text']['base_dir'])
    medical_path = base_data_dir / "medical_data.jsonl"
    if medical_path.exists():
        rows = load_jsonl_lines(medical_path)
        texts, metas = [], []
        for obj in tqdm(rows, desc="Processing medical data"):
            text = medical_to_text(obj)
            # medical 条目通常较长，也切一下更好召回
            chunks = split_text(text, chunk_size=900, overlap_size=120)
            for ci, c in enumerate(chunks):
                texts.append(c)
                metas.append(sanitize_metadata({
                    "category": "chunyuyisheng_medical",
                    "department": obj.get("department"),
                    "disease_name": obj.get("disease_name"),
                    "url": obj.get("url")
                }))
        knowledge_db.add_texts(texts, metas)
        logger.info(f"medical_data.jsonl ingested into knowledge: {len(rows)} rows")
    else:
        logger.warning("medical_data.jsonl not found, skip")
    # 3) redflag: red_flag.jsonl（一行一个doc，通常不切）
    redflag_path = base_data_dir / "red_flag.jsonl"
    if redflag_path.exists():
        rows = load_jsonl_lines(redflag_path)
        texts, metas, ids = [], [], []
        for obj in tqdm(rows, desc="Processing redflag data"):
            texts.append(redflag_to_text(obj))
            metas.append(sanitize_metadata({
                "category": "redflag",
                "id": obj.get("id")
            }))
            # 用你自己的 id 更利于去重
            ids.append(obj.get("id") or str(uuid.uuid4()))
        redflag_db.add_texts(texts, metas, ids=ids)
        logger.info(f"red_flag.jsonl ingested into redflag: {len(rows)} rows")
    else:
        logger.warning("red_flag.jsonl not found, skip")
    # 4) triage: triagle.jsonl（一行一个doc，通常不切）
    triage_path = base_data_dir / "triagle.jsonl"
    if triage_path.exists():
        rows = load_jsonl_lines(triage_path)
        texts, metas = [], []
        for obj in tqdm(rows, desc="Processing triage data"):
            texts.append(triage_to_text(obj))
            metas.append(sanitize_metadata({
                "category": "triage",
                "symptom": obj.get("symptom"),
            }))
        triage_db.add_texts(texts, metas)
        logger.info(f"triagle.jsonl ingested into triage: {len(rows)} rows")
    else:
        logger.warning("triagle.jsonl not found, skip")
    logger.info("All transfer_data collections built successfully.")


if __name__ == "__main__":
    # reset=True 会清空原collection重新灌库，第一次建议 True，后续增量可以 False
    build_transfer_data_vectorstores(reset=False)