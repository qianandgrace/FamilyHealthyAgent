import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import random
import json
from openai import OpenAI
import os
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv("LAOZHANG_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://api.laozhang.ai/v1"  # 百炼服务的base_url
)

BASE = "https://dxy.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://dxy.com"
}

# =========================
# 1. 获取科室列表
# =========================
def get_departments():
    url = "https://dxy.com/diseases"
    html = requests.get(url, headers=HEADERS).text
    soup = BeautifulSoup(html, "html.parser")

    departments = []
    for a in soup.select("a"):
        href = a.get("href", "")
        name = a.text.strip()

        if "/diseases/" in href and name!="查疾病":
            departments.append({
                "name": name,
                "url": urljoin(BASE, href)
            })

    return departments


# =========================
# 2. 获取疾病URL
# =========================
def extract_diseases_json(html):
    start_key = '"diseases":['
    start = html.find(start_key)

    if start == -1:
        return []

    start += len('"diseases":')

    bracket_count = 0
    end = start

    for i in range(start, len(html)):
        if html[i] == '[':
            bracket_count += 1
        elif html[i] == ']':
            bracket_count -= 1

            if bracket_count == 0:
                end = i + 1
                break

    json_str = html[start:end]

    return json.loads(json_str)[0]


def get_disease_urls(dept):
    html = requests.get(dept["url"], headers=HEADERS).text
    disease_block = extract_diseases_json(html)

    disease_list = []
    for tag in disease_block.get("tag_list", []):
        tag_id = tag["tag_id"]
        name = tag["tag_name"]

        disease_list.append({
            "department": dept["name"],
            "name": name,
            "url": f"https://dxy.com/disease/{tag_id}/detail"
        })

    return disease_list


# =========================
# 3. 解析疾病详情页（核心🔥）
# =========================
def parse_sections(soup):
    sections = {
        "symptoms": "",
        "causes": "",
        "treatment": "",
        "when_to_hospital": ""
    }

    # 找所有标题块
    headers = soup.find_all(["h2", "h3"])

    for h in headers:
        title = h.text.strip()

        content = []
        for sib in h.find_next_siblings():
            # 遇到下一个标题就停
            if sib.name in ["h2", "h3"]:
                break

            text = sib.get_text(strip=True)
            if text:
                content.append(text)

        content_text = "\n".join(content)

        # 分类
        if any(k in title for k in ["症状", "表现"]):
            sections["symptoms"] = content_text
        elif any(k in title for k in ["病因", "原因"]):
            sections["causes"] = content_text
        elif any(k in title for k in ["治疗", "用药"]):
            sections["treatment"] = content_text
        elif any(k in title for k in ["就医", "医院", "何时"]):
            sections["when_to_hospital"] = content_text

    return sections


def parse_disease_detail(disease):
    url = disease["url"]

    try:
        html = requests.get(url, headers=HEADERS, timeout=10).text
    except:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # 标题
    title = soup.find("h1")
    disease_name = title.text.strip() if title else disease["name"]

    # ✅ 用结构化解析
    sections = parse_sections(soup)

    # 如果解析失败，fallback
    if not any(sections.values()):
        content = soup.get_text()
        sections["symptoms"] = content[:500]

    # ✅ 红旗症状
    RED_FLAGS = ["抽搐", "昏迷", "呼吸困难", "持续高烧"]
    full_text = " ".join(sections.values())

    red_flags = [f for f in RED_FLAGS if f in full_text]

    return {
        "department": disease["department"],
        "disease_name": disease_name,
        "url": url,
        "symptoms": sections["symptoms"],
        "causes": sections["causes"],
        "treatment": sections["treatment"],
        "when_to_hospital": sections["when_to_hospital"],
        "red_flags": red_flags
    }

def enrich_with_llm(data):
    prompt = f"""
你是一个医疗知识结构化专家，请基于以下信息补充：

疾病：{data['disease_name']}

症状：
{data['symptoms']}

要求输出JSON：
1. risk_level: low / medium / high
2. triage: home / clinic / emergency
3. red_flags: 补充关键危险信号（数组）
4. summary: 200字以内总结（重写）

注意：
- 不要编造具体用药剂量
- 保守判断（医疗安全优先）
"""

    try:
        response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # 更稳定
    )
        result = response.choices[0].message.content.strip()

        return safe_json_loads(result)
    except:
        return {}
    
def safe_json_loads(text):
    import re
    import json

    try:
        return json.loads(text)
    except:
        pass

    # fallback：提取 JSON
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None

# =========================
# 4. 主流程
# =========================
def load_processed():
    if not os.path.exists("processed.txt"):
        return set()
    with open("processed.txt") as f:
        return set(line.strip() for line in f)

def save_processed(url):
    with open("processed.txt", "a") as f:
        f.write(url + "\n")

def run():

    departments = get_departments()

    for dept in tqdm(departments):
        print(f"\n📂 科室: {dept['name']}")

        diseases = get_disease_urls(dept)
        processed = load_processed()
        for d in diseases[:20]:  # 控制量
            if d["url"] in processed:
                continue
            print("  🦠", d["name"])

            detail = parse_disease_detail(d)

            if detail:
                llm_data = enrich_with_llm(detail)
                if llm_data:
                    detail.update(llm_data)
                # 保存
                with open("medical_data.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
                save_processed(d["url"])
            time.sleep(random.uniform(1, 2))


if __name__ == "__main__":
    run()