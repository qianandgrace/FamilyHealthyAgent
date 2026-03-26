import logging
import re
import json
import os
from typing import Dict, Any
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 当处理中文文本时，按照标点进行断句
def sent_tokenize(input_string):
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # 去掉空字符串
    return [sentence for sentence in sentences if sentence.strip()]


def split_text(paragraphs, chunk_size=800, overlap_size=200):
    # 按指定 chunk_size 和 overlap_size 交叠割文本
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev_len = 0
        prev = i - 1
        # 向前计算重叠部分
        while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap+chunk
        next = i + 1
        # 向后计算当前chunk
        while next < len(sentences) and len(sentences[next])+len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    # logger.info(f"chunks: {chunks[0:10]}")
    return chunks

# ---------- JSONL -> text ----------
def medical_to_text(obj: Dict[str, Any]) -> str:
    # 把你 medical_data.jsonl 的核心字段拼成更好检索的“知识条目”
    red_flags = obj.get("red_flags") or []
    rf_txt = "；".join(red_flags) if isinstance(red_flags, list) else str(red_flags)
    return "\n".join([
        f"科室: {obj.get('department','')}",
        f"疾病: {obj.get('disease_name','')}",
        f"概述: {obj.get('summary','')}",
        f"症状: {obj.get('symptoms','')}",
        f"病因: {obj.get('causes','')}",
        f"治疗: {obj.get('treatment','')}",
        f"何时就医: {obj.get('when_to_hospital','')}",
        f"红旗信号: {rf_txt}",
        f"风险等级: {obj.get('risk_level','')}",
        f"分诊建议: {obj.get('triage','')}",
        f"来源: {obj.get('url','')}",
    ]).strip()


def redflag_to_text(obj: Dict[str, Any]) -> str:
    kws = obj.get("keywords") or []
    kw_txt = "、".join(kws) if isinstance(kws, list) else str(kws)
    return "\n".join([
        f"红旗ID: {obj.get('id','')}",
        f"危险情况: {obj.get('condition','')}",
        f"关键词: {kw_txt}",
        f"风险等级: {obj.get('risk_level','')}",
        f"建议动作: {obj.get('action','')}",
        f"适用年龄: {obj.get('age_group','')}",
        f"适用性别: {obj.get('sex','')}",
        f"时间窗: {obj.get('time_window', '')}",
        f"备注: {obj.get('notes','')}",
    ]).strip()


def triage_to_text(obj: Dict[str, Any]) -> str:
    rules = obj.get("triage_rules") or []
    rules_lines = []
    for r in rules:
        rules_lines.append(
            f"- 条件: {r.get('condition','')} => level={r.get('level','')} action={r.get('action','')}"
        )
    rules_txt = "\n".join(rules_lines)
    return "\n".join([
        f"症状: {obj.get('symptom','')}",
        f"人群: {obj.get('population','')}",
        "分诊规则:",
        rules_txt
    ]).strip()

# ---------- Loaders ----------
def load_md_as_chunks(md_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    chunks = split_text(text, chunk_size=800, overlap_size=200)
    texts, metas = [], []
    for i, c in enumerate(chunks):
        texts.append(c)
        metas.append({
            "category": "weijianwei_knowledge",
            "format": "md",
            "source": str(md_path),
            "chunk": i,
        })
    return texts, metas


def load_jsonl_lines(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_line"] = i
                rows.append(obj)
            except Exception as e:
                logger.warning(f"Bad json at {jsonl_path} line {i}: {e}")
    return rows


if __name__ == "__main__":
    # md file loader test
    md_dir = "data\\transfer_data"
    md_dir = Path(md_dir)
    md_files = list(md_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} md files")
    for p in md_files:
        breakpoint()
        texts, metas = load_md_as_chunks(p)
