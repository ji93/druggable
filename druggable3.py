#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import ssl
import certifi
import requests
import pandas as pd
import xml.etree.ElementTree as ET

from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ============================================================
# SSL 修复
# ============================================================
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
except Exception as e:
    print(f"⚠️ SSL 配置警告: {e}")

# ============================================================
# 依赖检查
# ============================================================
try:
    from dashscope import Generation
    HAS_DASHSCOPE = True
except ImportError:
    print("⚠️ 未安装 dashscope，所有 LLM 角色降级为规则模式")
    HAS_DASHSCOPE = False

try:
    from Bio import Entrez
    HAS_BIOPYTHON = True
except ImportError:
    print("⚠️ 未安装 biopython：pip install biopython")
    HAS_BIOPYTHON = False

DASHSCOPE_API_KEY = [DASHSCOPE_API_KEY]
MODEL = "qwen-plus"
Entrez.email = "xxx"


# ============================================================
# 1. LLM 基础调用
# ============================================================
def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            fixed = _try_fix_truncated_json(candidate)
            if fixed is not None:
                return fixed
    raise ValueError(f"No valid JSON: {text[:200]}")


def _try_fix_truncated_json(s: str):
    depth_curly = depth_square = 0
    in_str = False
    esc = False
    last_safe = 0
    for i, ch in enumerate(s):
        if esc: esc = False; continue
        if ch == "\\": esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if in_str: continue
        if ch == "{": depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
            if depth_curly == 0 and depth_square == 0:
                last_safe = i + 1
        elif ch == "[": depth_square += 1
        elif ch == "]": depth_square -= 1

    if last_safe > 0:
        try: return json.loads(s[:last_safe])
        except Exception: pass

    tail = ('"' if in_str else "") + "]" * max(depth_square, 0) + "}" * max(depth_curly, 0)
    try: return json.loads(s + tail)
    except Exception: return None


def call_llm(prompt: str, system_message: str = None,
             temperature: float = 0.0,
             enable_search: bool = False,
             timeout: int = 40) -> Optional[Dict]:
    if not HAS_DASHSCOPE or not DASHSCOPE_API_KEY:
        return None

    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': prompt})

    try:
        kwargs = dict(
            model=MODEL,
            messages=messages,
            api_key=DASHSCOPE_API_KEY,
            result_format='message',
            temperature=temperature,
            timeout=timeout,
        )
        if enable_search:
            kwargs['enable_search'] = True

        response = Generation.call(**kwargs)
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            return extract_json(content)
        print(f"   [call_llm] ❌ status={response.status_code} "
              f"code={getattr(response, 'code', '?')} "
              f"message={getattr(response, 'message', '?')}")
        return None
    except Exception as e:
        print(f"   [call_llm] ❌ {type(e).__name__}: {e}")
        return None


# ============================================================
# 2. Agent 角色 system prompts
# ============================================================
AGENT1_SYSTEM = """你是药物化学家，评估靶点的可药性 (druggability)。
你会收到 Open Targets tractability 原始证据。

【打分】
- 可药性评分最高100，最低0，由你根据该基因的可药性证据，自主评估其在可药性程度。

【输出】
只返回 JSON：
{
  "score": 0-100,
  "evidence": {"key_findings": ["..."], "raw_signals": {"pocket": bool, "family": bool, "binder": bool}},
  "concerns": ["..."],
  "confidence": 0.0-1.0
}
"""

AGENT2_SYSTEM = """你是通路生物学家，评估基因在心脏纤维化核心信号网络中的地位。
你会收到该基因在 KEGG 中命中的所有通路。

核心纤维化通路包括：
TGF-beta / MAPK / PI3K-Akt / Focal adhesion / ECM-receptor /
Hippo / TNF / AGE-RAGE / 各型 cardiomyopathy / Viral myocarditis

【打分规则】
- 由你根据该基因命中的 KEGG 通路，自主评估其在心脏纤维化核心信号网络中的地位。

【输出】
只返回 JSON：
{
  "score": 0-100,
  "evidence": {"core_pathways": ["..."], "network_role": "hub/intermediate/peripheral"},
  "concerns": ["..."],
  "confidence": 0.0-1.0
}
"""

AGENT3_SYSTEM = """你是心血管病理学家。你会收到关于某个基因的若干篇 PubMed 文献的摘要级证据，
每篇都已被抽取为 JSON，包含：is_relevant / fibrosis_related /
species / directionality / cell_types / mechanisms / novelty。

【评分规则】
- 综合评价该基因与心脏纤维化的文献证据质量，0-100分。
- 基因表达与纤维化的正相关关系，是加分的硬性指标和优先依据。

【输出】
只返回 JSON：
{
  "score": 0-100,
  "evidence": {
     "n_papers": int,
     "n_relevant": int,
     "dominant_direction": "up/down/conflicting/unknown",
     "dominant_species": ["human", "mouse"],
     "dominant_cell_types": ["cardiac fibroblasts"],
     "top_mechanisms": ["TGF-beta/SMAD", "..."],
     "key_findings": ["..."]
  },
  "concerns": ["..."],
  "confidence": 0.0-1.0
}
"""

ORCHESTRATOR_SYSTEM_PROMPT = """你是心脏纤维化靶点筛选流水线的调度中枢。
你控制三阶段：Agent1 可药性 → Agent2 通路关键性 → Agent3 文献证据。
每个阶段你会收到该 Agent 的 LLM 结论 + 原始证据。

【Agent1 通过规则】
- 命中 High-Quality Pocket / Druggable Family，进入 Agent2
- 否则淘汰

【Agent2 通过规则】
- 核心通路命中 >= 1，进入 Agent3
- 否则淘汰

【Agent3 通过规则】
- n_relevant >= 1，即至少一篇相关文献，进入最终反思
- 否则淘汰

【输出】
只返回 JSON：
{
  "decision": "pass" | "fail",
  "reason": "一句话中文理由",
  "score_adjustment": 0,
  "next_action": "continue_agent2" | "continue_agent3" | "finalize" | "drop"
}
"""

REVIEWER_SYSTEM = """你是跨学科评审员，审查三个 Agent 对同一基因的结论是否矛盾、是否过度自信。

【常见矛盾模式】
- 可药性高但文献/通路证据薄弱
- 文献证据强但方向冲突（up vs down）
- 通路命中多但都是泛通路

【输出】
只返回 JSON：
{
  "agreement": "high" | "medium" | "low",
  "conflicts": [{"between": "agent1 vs agent2", "issue": "..."}],
  "overall_comment": "一句话总评",
  "suggested_adjustment": 0
}
"""

EVIDENCE_EXTRACTOR_SYSTEM = """你是文献证据抽取专家，专门从 PubMed 摘要中提取结构化信息。

【任务】
给定标题 + 摘要，判断并抽取以下字段：
- is_relevant: 是否与心脏纤维化直接相关（必须同时包含 heart/cardiac 与 fibrosis/fibrotic/collagen/scar）
- fibrosis_related: 是否涉及纤维化
- species: 涉及物种（human / mouse / rat / monkey / in_vitro / unknown），可多选
- directionality: 该基因在纤维化中的方向（up / down / no_change / conflicting / unknown）
- cell_types: 涉及的细胞类型（如 cardiac fibroblasts / cardiomyocytes / endothelial cells）
- mechanisms: 涉及的机制/通路（如 TGF-beta/SMAD、MAPK、YAP/TAZ、ECM remodeling）
- key_finding: 一句话核心发现
- novelty: 该文献提出的新点（0-1，无新意=0.1，全新机制=0.9）
- confidence: 你对该抽取结果的信心（0-1）

【规则】
- 只依据给出的标题和摘要，不要编造
- 若摘要信息不足，species/directionality/cell_types 等返回 "unknown" 或 []
- 必须返回严格 JSON，无额外文本

【输出示例】
{
  "is_relevant": true,
  "fibrosis_related": true,
  "species": ["mouse"],
  "directionality": "up",
  "cell_types": ["cardiac fibroblasts"],
  "mechanisms": ["TGF-beta/SMAD", "ECM remodeling"],
  "key_finding": "TGFB1 在 MI 后小鼠心脏成纤维细胞中上调并促进胶原沉积",
  "novelty": 0.4,
  "confidence": 0.85
}
"""


# ============================================================
# 3. Open Targets 客户端
# ============================================================
class OpenTargetsClient:
    GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _post(self, query: str, variables: dict = None) -> dict:
        payload = {"query": query, "variables": variables or {}}
        resp = self.session.post(self.GRAPHQL_URL, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def search_entities(self, query: str) -> List[dict]:
        q = """
        query SearchEntities($query: String!) {
          search(query: $query, entityTypes: [TARGET], first: 5) {
            edges { node { id name entityType
              ... on Target { ensemblId approvedSymbol approvedName species { name } } } }
          }
        }"""
        data = self._post(q, {"query": query})
        return [e["node"] for e in data.get("data", {}).get("search", {}).get("edges", [])]

    def get_tractability(self, ensembl_id: str) -> dict:
        q = """
        query TargetTractability($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            approvedSymbol
            tractability { modality label value }
          }
        }"""
        data = self._post(q, {"ensemblId": ensembl_id})
        return data.get("data", {}).get("target", {})

    def get_ensembl_id(self, gene_symbol: str) -> Optional[str]:
        results = self.search_entities(gene_symbol)
        for n in results:
            if (n.get("species", {}).get("name") == "human" and
                    n.get("approvedSymbol", "").upper() == gene_symbol.upper()):
                return n.get("ensemblId")
        for n in results:
            if n.get("ensemblId"):
                return n.get("ensemblId")
        return None


# ============================================================
# 4. KEGG
# ============================================================
class KeggAgent:
    def __init__(self):
        self.base_url = "https://rest.kegg.jp"
        self.cache = {}

    def get_pathways(self, gene: str) -> Dict:
        gu = gene.upper()
        if gu in self.cache:
            return self.cache[gu]
        try:
            resp = requests.get(f"{self.base_url}/get/hsa:{gu}", timeout=15)
            pathways = []
            if resp.status_code == 200:
                in_pathway = False
                for line in resp.text.split('\n'):
                    if line.startswith('PATHWAY'):
                        in_pathway = True
                        continue
                    if in_pathway and line.startswith(' '):
                        parts = line.strip().split('  ')
                        if len(parts) >= 2:
                            pathways.append({"id": parts[0], "name": parts[1]})
                    else:
                        in_pathway = False
            result = {"gene": gu, "pathways": pathways, "total_pathways": len(pathways)}
            self.cache[gu] = result
            return result
        except Exception as e:
            print(f"   ⚠️ KEGG error: {e}")
            return {"gene": gu, "pathways": [], "total_pathways": 0}


# ============================================================
# 5. PubMed 检索 + 去重 + 摘要级 evidence extraction
# ============================================================
class PubMedEvidencePipeline:
    QUERY_TEMPLATES = [
        '({gene}) AND (fibroblast activation)',
        '({gene}) AND (fibrosis)',
        '({gene}) AND (TGF-beta OR TGFB1 OR TGFβ)',
        '({gene}) AND (myofibroblast differentiation OR EMT)',
    ]

    def __init__(self, max_papers_per_gene: int = 40):
        self.max_papers = max_papers_per_gene
        self.cache = {}

    def search_pmids(self, gene: str) -> List[str]:
        if not HAS_BIOPYTHON:
            return []
        pmids = set()
        for tpl in self.QUERY_TEMPLATES:
            q = tpl.format(gene=gene)
            try:
                handle = Entrez.esearch(db="pubmed", term=q, retmax=20, sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                for p in record.get("IdList", []):
                    pmids.add(str(p))
            except Exception as e:
                print(f"   ⚠️ esearch error ({q}): {e}")
            time.sleep(0.3)
            if len(pmids) >= self.max_papers:
                break
        return list(pmids)[: self.max_papers]

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        if not pmids:
            return []
        papers = []
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        for i in range(0, len(pmids), 20):
            chunk = pmids[i:i+20]
            params = {
                "db": "pubmed",
                "id": ",".join(chunk),
                "retmode": "xml",
                "rettype": "abstract",
                "email": Entrez.email,
            }
            try:
                r = requests.get(base, params=params, timeout=30)
                r.raise_for_status()
                root = ET.fromstring(r.content)

                for art in root.findall(".//PubmedArticle"):
                    pmid_el = art.find(".//PMID")
                    pmid = pmid_el.text if pmid_el is not None else ""

                    title_el = art.find(".//ArticleTitle")
                    title = "".join(title_el.itertext()) if title_el is not None else ""

                    ab_parts = []
                    for ab in art.findall(".//Abstract/AbstractText"):
                        label = ab.get("Label")
                        text = "".join(ab.itertext())
                        if label:
                            ab_parts.append(f"{label}: {text}")
                        else:
                            ab_parts.append(text)
                    abstract = " ".join(ab_parts)

                    j_el = art.find(".//Journal/Title")
                    journal = j_el.text if j_el is not None else ""

                    year = ""
                    y_el = art.find(".//PubDate/Year")
                    if y_el is not None:
                        year = y_el.text or ""
                    else:
                        md = art.find(".//PubDate/MedlineDate")
                        if md is not None and md.text:
                            m = re.search(r"(\d{4})", md.text)
                            if m:
                                year = m.group(1)

                    papers.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "year": year,
                    })
            except Exception as e:
                print(f"   ⚠️ efetch error: {e}")
            time.sleep(0.4)

        return papers

    def extract_evidence(self, paper: Dict, gene: str) -> Optional[Dict]:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        if not abstract:
            return None
        prompt = f"""目标基因: {gene}
标题: {title}
摘要: {abstract[:2000]}

请按 system prompt 要求抽取证据，只返回 JSON。"""
        res = call_llm(prompt, EVIDENCE_EXTRACTOR_SYSTEM, temperature=0.0)
        if res and isinstance(res, dict):
            res.setdefault("pmid", paper.get("pmid"))
            res.setdefault("title", title)
            res.setdefault("journal", paper.get("journal"))
            res.setdefault("year", paper.get("year"))
            return res
        return self._rule_extract(paper, gene)

    def _rule_extract(self, paper: Dict, gene: str) -> Dict:
        text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
        heart = any(k in text for k in ["cardiac", "heart", "myocardial", "ventricular"])
        fib = any(k in text for k in ["fibrosis", "fibrotic", "collagen", "ecm", "scar"])
        species = []
        if "mouse" in text or "mice" in text: species.append("mouse")
        if "rat" in text: species.append("rat")
        if "human" in text or "patient" in text: species.append("human")
        if "monkey" in text or "macaque" in text: species.append("monkey")
        direction = "unknown"
        if re.search(r'\bupregulat|\bincreas|\belevat', text): direction = "up"
        if re.search(r'\bdownregulat|\bdecreas|\breduc', text): direction = "down"
        return {
            "is_relevant": heart and fib,
            "fibrosis_related": fib,
            "species": species or ["unknown"],
            "directionality": direction,
            "cell_types": [],
            "mechanisms": [],
            "key_finding": "[rule-based fallback]",
            "novelty": 0.2,
            "confidence": 0.5,
            "pmid": paper.get("pmid"),
            "title": paper.get("title"),
            "journal": paper.get("journal"),
            "year": paper.get("year"),
        }

    def run(self, gene: str, verbose: bool = True) -> Dict:
        if gene in self.cache:
            return self.cache[gene]

        if verbose:
            print(f"   📚 PubMed 检索: {gene}")
        pmids = self.search_pmids(gene)
        if verbose:
            print(f"   📚 检索到 {len(pmids)} 篇（去重后）")

        papers = self.fetch_abstracts(pmids)
        if verbose:
            print(f"   📚 获取到 {len(papers)} 篇摘要")

        evidences = []
        for i, p in enumerate(papers, 1):
            ev = self.extract_evidence(p, gene)
            if ev:
                evidences.append(ev)
            if i % 5 == 0:
                time.sleep(0.2)

        n_papers = len(papers)
        n_relevant = sum(1 for e in evidences if e.get("is_relevant"))
        directions = [e.get("directionality", "unknown") for e in evidences if e.get("is_relevant")]
        species_set = set()
        for e in evidences:
            if e.get("is_relevant"):
                for s in e.get("species", []):
                    species_set.add(s)
        cell_types_set = set()
        mech_set = set()
        for e in evidences:
            if e.get("is_relevant"):
                for ct in e.get("cell_types", []) or []:
                    cell_types_set.add(ct)
                for m in e.get("mechanisms", []) or []:
                    mech_set.add(m)

        result = {
            "gene": gene,
            "n_papers": n_papers,
            "n_relevant": n_relevant,
            "directions": directions,
            "species": list(species_set),
            "cell_types": list(cell_types_set),
            "mechanisms": list(mech_set),
            "evidences": evidences,
        }
        self.cache[gene] = result
        return result


# ============================================================
# 6. 三个 Agent + 规则兜底
# ============================================================
class ProteinScorer:
    CORE_FIBROSIS_PATHWAYS = {
        "hsa04350": "TGF-beta signaling pathway",
        "hsa04010": "MAPK signaling pathway",
        "hsa04151": "PI3K-Akt signaling pathway",
        "hsa04510": "Focal adhesion",
        "hsa04512": "ECM-receptor interaction",
        "hsa04390": "Hippo signaling pathway",
        "hsa04668": "TNF signaling pathway",
        "hsa04933": "AGE-RAGE signaling pathway",
        "hsa05410": "Hypertrophic cardiomyopathy",
        "hsa05412": "Arrhythmogenic right ventricular cardiomyopathy",
        "hsa05414": "Dilated cardiomyopathy",
        "hsa05415": "Diabetic cardiomyopathy",
        "hsa05416": "Viral myocarditis",
    }

    def __init__(self):
        self.ot = OpenTargetsClient()
        self.kegg = KeggAgent()
        self.pubmed = PubMedEvidencePipeline(max_papers_per_gene=40)

        self.ensembl_map = {
            "MYH7": "ENSG00000092054", "MYH6": "ENSG00000197616",
            "MYH9": "ENSG00000100345", "MYH10": "ENSG00000133026",
            "MYH4": "ENSG00000197561", "VIM": "ENSG00000026025",
            "DES": "ENSG00000175084", "ACTA2": "ENSG00000107796",
            "ACTB": "ENSG00000075624", "GAPDH": "ENSG00000111640",
            "TUBB5": "ENSG00000196230", "TUBB4B": "ENSG00000188229",
            "TUBB2A": "ENSG00000137267", "TUBB2B": "ENSG00000137285",
            "TUBA1A": "ENSG00000167552", "ACTG1": "ENSG00000184009",
            "MYH11": "ENSG00000133392", "MYBPC3": "ENSG00000129991",
            "IMMT": "ENSG00000132326", "PFKP": "ENSG00000145632",
            "SORBS2": "ENSG00000154556", "HSP90AB1": "ENSG00000096384",
            "ATP5F1A": "ENSG00000152234", "CLTC": "ENSG00000141367",
            "ACTC1": "ENSG00000159251", "IDH2": "ENSG00000182054",
            "LDHA": "ENSG00000134333", "TUBA1C": "ENSG00000167553",
            "HSPA8": "ENSG00000109971", "TUBB3": "ENSG00000258947",
            "ATP5F1B": "ENSG00000110955", "LRPPRC": "ENSG00000138095",
            "UQCRC1": "ENSG00000010295", "HNRNPA2B1": "ENSG00000122566",
            "TUBA4A": "ENSG00000127824", "UQCRC2": "ENSG00000140740",
            "EIF3A": "ENSG00000107581", "ATP5PO": "ENSG00000124217",
            "HSPA5": "ENSG00000044574", "HADHA": "ENSG00000084754",
            "HSPB1": "ENSG00000106211", "TGM2": "ENSG00000198959",
            "ATP2A2": "ENSG00000174437", "SERPINH1": "ENSG00000149257",
            "KHSRP": "ENSG00000115266", "MYL3": "ENSG00000160808",
            "GSTM2": "ENSG00000213366", "PABPC1": "ENSG00000130695",
            "HSP90AA1": "ENSG00000080824", "ATP5PD": "ENSG00000167840",
            "EEF1A1": "ENSG00000156508", "HNRNPM": "ENSG00000099783",
            "FUBP1": "ENSG00000162613", "LMNA": "ENSG00000160789",
            "JUP": "ENSG00000173801", "STAT3": "ENSG00000168610",
            "MYH8": "ENSG00000133033", "HNRNPH2": "ENSG00000126945",
            "SAMM50": "ENSG00000119711", "HADHB": "ENSG00000138029",
            "ACAA2": "ENSG00000167315", "HMGB1": "ENSG00000189403",
            "PFKM": "ENSG00000152556", "ATP5PB": "ENSG00000166923",
            "HNRNPH1": "ENSG00000169045", "CSRP3": "ENSG00000129170",
            "ECI1": "ENSG00000149657", "HDLBP": "ENSG00000115677",
            "SRP54": "ENSG00000109158", "RPS5": "ENSG00000083845",
            "LDHB": "ENSG00000111716", "CSRP1": "ENSG00000159176",
            "PDLIM5": "ENSG00000163110", "FN1": "ENSG00000115414",
            "DARS1": "ENSG00000131580", "VAT1": "ENSG00000108828",
            "GSN": "ENSG00000148180", "HSPA9": "ENSG00000113013",
            "ACSL1": "ENSG00000151726", "CASQ2": "ENSG00000118729",
            "EEF1A2": "ENSG00000132676", "VARS1": "ENSG00000109184",
            "AIFM1": "ENSG00000156711", "PFKL": "ENSG00000141959",
            "EEF1D": "ENSG00000104529", "PTGIS": "ENSG00000124212",
            "COL3A1": "ENSG00000168542", "PICALM": "ENSG00000073921",
            "WARS1": "ENSG00000140105", "MAP4": "ENSG00000047849",
            "NDUFS1": "ENSG00000023228", "RPN1": "ENSG00000163902",
            "ILK": "ENSG00000166333", "ATAD3": "ENSG00000197785",
            "SLC25A13": "ENSG00000104864", "TRAP1": "ENSG00000126602",
            "EEF1G": "ENSG00000132383", "RARS1": "ENSG00000113643",
            "LMOD2": "ENSG00000109686", "SLC25A12": "ENSG00000115840",
            "RTCB": "ENSG00000100314", "GSTM4": "ENSG00000168615",
            "PSMD11": "ENSG00000108821", "PGK1": "ENSG00000102144",
            "RPS7": "ENSG00000171863", "RAN": "ENSG00000132341",
            "TPM2": "ENSG00000198467", "TNNT2": "ENSG00000118194",
            "ACSL3": "ENSG00000123983", "PRSS1": "ENSG00000204983",
        }

    def _get_ensembl(self, gene: str) -> Optional[str]:
        gu = gene.upper()
        if gu in self.ensembl_map:
            return self.ensembl_map[gu]
        try:
            return self.ot.get_ensembl_id(gene)
        except Exception:
            return None

    def fetch_druggability_raw(self, gene: str) -> Dict:
        raw = {"ensembl_id": None, "tractability_list": [], "error": None}
        ensembl = self._get_ensembl(gene)
        if not ensembl:
            raw["error"] = "No Ensembl ID"
            return raw
        raw["ensembl_id"] = ensembl
        try:
            target = self.ot.get_tractability(ensembl)
            raw["tractability_list"] = target.get("tractability", []) if target else []
            if not target:
                raw["error"] = "API returned empty"
        except Exception as e:
            raw["error"] = str(e)
        return raw

    def fetch_kegg_raw(self, gene: str) -> Dict:
        raw = {"total_pathways": 0, "pathways": [],
               "core_pathway_hits": 0, "core_pathways": [], "error": None}
        try:
            data = self.kegg.get_pathways(gene)
            pathways = data.get("pathways", [])
            raw["total_pathways"] = len(pathways)
            raw["pathways"] = pathways
            core = [{"id": p["id"], "name": self.CORE_FIBROSIS_PATHWAYS[p["id"]]}
                    for p in pathways if p["id"] in self.CORE_FIBROSIS_PATHWAYS]
            raw["core_pathway_hits"] = len(core)
            raw["core_pathways"] = core
        except Exception as e:
            raw["error"] = str(e)
        return raw

    @staticmethod
    def _rule_agent1(raw: Dict) -> Dict:
        score = 0
        pocket = family = binder = False
        sm = 0
        for item in [t for t in raw.get("tractability_list", []) if t.get("modality") == "SM"]:
            if not item.get("value"): continue
            label = item.get("label", "")
            if "Approved Drug" in label: sm += 40; binder = True
            elif "Advanced Clinical" in label: sm += 30; binder = True
            elif "Phase 1 Clinical" in label: sm += 20; binder = True
            elif "Structure with Ligand" in label: sm += 18
            elif "High-Quality Ligand" in label: sm += 15
            elif "High-Quality Pocket" in label: sm += 20; pocket = True
            elif "Druggable Family" in label: sm += 18; family = True
        score += min(sm, 65)
        ab = 0
        for item in [t for t in raw.get("tractability_list", []) if t.get("modality") == "AB"]:
            if not item.get("value"): continue
            label = item.get("label", "")
            if "Approved Drug" in label: ab += 30
            elif "Advanced Clinical" in label: ab += 25
            elif "Phase 1 Clinical" in label: ab += 15
            elif "UniProt loc" in label or "GO CC" in label: ab += 5
        score += min(ab, 30)
        pr = 0
        for item in [t for t in raw.get("tractability_list", []) if t.get("modality") == "PR"]:
            if not item.get("value"): continue
            label = item.get("label", "")
            if "Approved Drug" in label: pr += 25
            elif "Advanced Clinical" in label: pr += 20
            elif "Phase 1 Clinical" in label: pr += 15
            elif "Literature" in label or "Ubiquitination" in label: pr += 5
        score += min(pr, 25)
        return {
            "score": min(score, 100),
            "evidence": {"key_findings": ["[rule-based]"],
                         "raw_signals": {"pocket": pocket, "family": family, "binder": binder}},
            "concerns": ["LLM unavailable"],
            "confidence": 0.5,
        }

    # ← 改动：改为通路规则（原来叫 _rule_agent2 是文献）
    @staticmethod
    def _rule_agent2_pathway(raw: Dict) -> Dict:
        core = raw.get("core_pathway_hits", 0)
        return {
            "score": min(core * 20, 100),
            "evidence": {
                "core_pathways": [p["name"] for p in raw.get("core_pathways", [])],
                "network_role": "unknown",
            },
            "concerns": ["LLM unavailable"],
            "confidence": 0.5,
        }

    # ← 改动：改为文献规则（原来叫 _rule_agent3 是通路）
    @staticmethod
    def _rule_agent3_literature(pubmed_result: Dict) -> Dict:
        n_rel = pubmed_result.get("n_relevant", 0)
        dirs = pubmed_result.get("directions", [])
        dominant = "unknown"
        if dirs:
            up = sum(1 for d in dirs if d == "up")
            down = sum(1 for d in dirs if d == "down")
            if up > 0 and down == 0: dominant = "up"
            elif down > 0 and up == 0: dominant = "down"
            elif up > 0 and down > 0: dominant = "conflicting"
        score = min(n_rel * 8, 60) + (15 if dominant in ("up", "down") else 0)
        return {
            "score": min(score, 100),
            "evidence": {
                "n_papers": pubmed_result.get("n_papers", 0),
                "n_relevant": n_rel,
                "dominant_direction": dominant,
                "dominant_species": pubmed_result.get("species", []),
                "dominant_cell_types": pubmed_result.get("cell_types", []),
                "top_mechanisms": pubmed_result.get("mechanisms", [])[:5],
                "key_findings": [f"n_relevant={n_rel}"],
            },
            "concerns": ["LLM unavailable"],
            "confidence": 0.5,
        }

    # --------------------------------------------------------
    # Agent 1: 可药性
    # --------------------------------------------------------
    def agent1_druggability(self, gene: str, raw: Dict) -> Dict:
        prompt = f"""基因: {gene}
Open Targets tractability 原始证据:
{json.dumps(raw, ensure_ascii=False, indent=2)}

以药物化学家身份打分并返回 JSON。"""
        res = call_llm(prompt, AGENT1_SYSTEM, temperature=0.0)
        if res and isinstance(res, dict) and "score" in res:
            res.setdefault("evidence", {})
            res.setdefault("concerns", [])
            res.setdefault("confidence", 0.7)
            return res
        return self._rule_agent1(raw)

    # --------------------------------------------------------
    # ← 改动：Agent 2 现在是通路
    # --------------------------------------------------------
    def agent2_pathway(self, gene: str, raw: Dict) -> Dict:
        prompt = f"""基因: {gene}
KEGG 通路:
{json.dumps(raw, ensure_ascii=False, indent=2)}

以通路生物学家身份打分并返回 JSON。"""
        res = call_llm(prompt, AGENT2_SYSTEM, temperature=0.0)
        if res and isinstance(res, dict) and "score" in res:
            res.setdefault("evidence", {})
            res.setdefault("concerns", [])
            res.setdefault("confidence", 0.7)
            return res
        return self._rule_agent2_pathway(raw)

    # --------------------------------------------------------
    # ← 改动：Agent 3 现在是文献
    # --------------------------------------------------------
    def agent3_literature(self, gene: str, pubmed_result: Dict) -> Dict:
        evidences = pubmed_result.get("evidences", [])
        relevant_evs = [e for e in evidences if e.get("is_relevant")]
        top_evs = relevant_evs[:10] if relevant_evs else evidences[:5]

        payload = {
            "gene": gene,
            "n_papers": pubmed_result.get("n_papers", 0),
            "n_relevant": pubmed_result.get("n_relevant", 0),
            "directions": pubmed_result.get("directions", []),
            "species": pubmed_result.get("species", []),
            "cell_types": pubmed_result.get("cell_types", []),
            "mechanisms": pubmed_result.get("mechanisms", []),
            "top_evidences": top_evs,
        }
        prompt = f"""基因: {gene}

PubMed 摘要级证据抽取汇总:
{json.dumps(payload, ensure_ascii=False, indent=2)}

以心血管病理学家身份综合评估并返回 JSON。"""
        res = call_llm(prompt, AGENT3_SYSTEM, temperature=0.0)
        if res and isinstance(res, dict) and "score" in res:
            res.setdefault("evidence", {})
            res.setdefault("concerns", [])
            res.setdefault("confidence", 0.7)
            return res
        return self._rule_agent3_literature(pubmed_result)


# ============================================================
# 7. 调度中枢 + Reviewer
# ============================================================
# ← 改动：调度顺序改为 可药性 → 通路 → 文献
def _rule_based_decision(stage: str, agent_result: Dict, raw: Dict) -> Dict:
    score = agent_result.get("score", 0)
    if stage == "agent1":
        # Agent1 可药性：命中 High-Quality Pocket / Druggable Family 才放行
        signals = agent_result.get("evidence", {}).get("raw_signals", {})
        pocket = signals.get("pocket", False)
        family = signals.get("family", False)
        passed = pocket or family
        return {
            "decision": "pass" if passed else "fail",
            "reason": (f"命中" + ("High-Quality Pocket " if pocket else "")
                       + ("Druggable Family " if family else "")).strip()
                      or f"可药性{score:.0f}分但未命中关键特征",
            "score_adjustment": 0,
            "next_action": "continue_agent2" if passed else "drop",
        }
    if stage == "agent2":
        # Agent2 通路：核心通路命中 >= 1
        core = raw.get("core_pathway_hits", 0)
        passed = core >= 1
        return {
            "decision": "pass" if passed else "fail",
            "reason": f"命中 {core} 个核心纤维化通路",
            "score_adjustment": 0,
            "next_action": "continue_agent3" if passed else "drop",
        }
    if stage == "agent3":
        # Agent3 文献：n_relevant >= 1
        n_rel = raw.get("n_relevant", 0)
        passed = n_rel >= 1
        return {
            "decision": "pass" if passed else "fail",
            "reason": f"相关文献 {n_rel} 篇",
            "score_adjustment": 0,
            "next_action": "finalize" if passed else "drop",
        }
    return {"decision": "fail", "reason": "unknown", "score_adjustment": 0, "next_action": "drop"}


def llm_orchestrate(stage: str, gene: str, agent_result: Dict, raw: Dict) -> Dict:
    fallback = _rule_based_decision(stage, agent_result, raw)
    if not HAS_DASHSCOPE or not DASHSCOPE_API_KEY:
        return fallback
    prompt = f"""阶段: {stage}
基因: {gene}

Agent 结论:
{json.dumps(agent_result, ensure_ascii=False, indent=2)[:3000]}

原始证据摘要:
{json.dumps(raw, ensure_ascii=False, indent=2, default=str)[:3000]}

依据 system prompt 规则返回 JSON。"""
    res = call_llm(prompt, ORCHESTRATOR_SYSTEM_PROMPT, temperature=0.0)
    if res and isinstance(res, dict) and res.get("decision") in ("pass", "fail"):
        return res
    return fallback


def cross_review(gene: str, a1: Dict, a2: Optional[Dict], a3: Optional[Dict]) -> Dict:
    if not HAS_DASHSCOPE or not DASHSCOPE_API_KEY:
        return {"agreement": "unknown", "conflicts": [],
                "overall_comment": "LLM unavailable", "suggested_adjustment": 0}
    # ← 改动：明确注明 a2=通路, a3=文献
    bundle = {"gene": gene,
              "agent1_druggability": a1,
              "agent2_pathway": a2,
              "agent3_literature": a3}
    prompt = f"""请审查三个 Agent 对 {gene} 的结论，找出矛盾或过度自信：

{json.dumps(bundle, ensure_ascii=False, indent=2, default=str)[:4000]}

只返回 JSON。"""
    res = call_llm(prompt, REVIEWER_SYSTEM, temperature=0.0)
    if res and isinstance(res, dict):
        res.setdefault("agreement", "unknown")
        res.setdefault("conflicts", [])
        res.setdefault("overall_comment", "")
        res.setdefault("suggested_adjustment", 0)
        return res
    return {"agreement": "unknown", "conflicts": [],
            "overall_comment": "review failed", "suggested_adjustment": 0}


def llm_reflect(top_candidates: List[Dict]) -> Dict:
    if not HAS_DASHSCOPE or not DASHSCOPE_API_KEY:
        return {"flagged_genes": [],
                "overall_assessment": "LLM 不可用",
                "recommended_top5": [g["gene"] for g in top_candidates[:5]]}

    summaries = []
    for idx, gene in enumerate(top_candidates[:20], 1):
        a1 = gene.get("agent1_result") or {}
        a2 = gene.get("agent2_result") or {}
        a3 = gene.get("agent3_result") or {}
        review = gene.get("cross_review") or {}
        ev3 = a3.get("evidence", {}) if a3 else {}
        # ← 改动：A2 现在显示通路，A3 显示文献
        summaries.append(
            f"{idx}. {gene['gene']} | 综合:{gene['overall']:.1f} | "
            f"A1可药:{a1.get('score', 0):.0f} | "
            f"A2通路:{a2.get('score', 0):.0f} | "
            f"A3文献:{a3.get('score', 0):.0f} (rel={ev3.get('n_relevant', 0)}, dir={ev3.get('dominant_direction', '?')}) | "
            f"互评:{review.get('agreement', '?')}"
        )

    prompt = f"""你是心脏纤维化药物靶点筛选专家。以下是通过三阶段（可药性→通路→文献）的候选基因：

{chr(10).join(summaries)}

请推荐 Top 5，并指出被低估或矛盾信号。

只返回 JSON：
{{"flagged_genes": [{{"gene": "xxx", "issue": "...", "action": "..."}}], "overall_assessment": "一句话总结", "recommended_top5": ["A","B","C","D","E"]}}
"""
    print(f"   [llm_reflect] prompt 长度: {len(prompt)} 字符")
    res = call_llm(prompt, "你只返回合法 JSON。", temperature=0.1)
    print(f"   [llm_reflect] LLM 返回: {res}")

    if res and isinstance(res, dict) and res.get("recommended_top5"):
        return res
    return {"flagged_genes": [],
            "overall_assessment": "LLM 无效，按综合分推荐",
            "recommended_top5": [g["gene"] for g in top_candidates[:5]]}


# ============================================================
# 8. 主程序
# ============================================================
def _pack_result(gene, a1, a2, a3, review, trace,
                 a1_raw, a2_raw, a3_raw, dropped_at=None, adj=0):
    s1 = a1.get("score", 0) if a1 else 0
    s2 = a2.get("score", 0) if a2 else 0
    s3 = a3.get("score", 0) if a3 else 0
    overall = 0.333 * s1 + 0.333 * s2 + 0.333 * s3 + adj
    return {
        "gene": gene,
        "druggability": round(s1, 2),
        "kegg_centrality": round(s2, 2),        # ← 改动：A2 现在是通路
        "literature_fibrosis": round(s3, 2),    # ← 改动：A3 现在是文献
        "overall": round(overall, 2),
        "score_adjustment": adj,
        "dropped_at": dropped_at,
        "agent1_result": a1,
        "agent2_result": a2,
        "agent3_result": a3,
        "agent1_raw": a1_raw,
        "agent2_raw": a2_raw,
        "agent3_raw": a3_raw,
        "cross_review": review,
        "trace": trace,
    }


def main():
    print("🧬 Role-specialized Multi-Agent 靶点筛选流水线 v3")
    print("   顺序：Agent1 可药性 → Agent2 通路 → Agent3 文献")
    print("=" * 70)

    if not DASHSCOPE_API_KEY:
        print("⚠️ 未检测到 DASHSCOPE_API_KEY，所有 Agent 走规则兜底")
    else:
        print("🔎 正在自检 LLM ...")
        test = call_llm('返回 JSON: {"status":"ok"}',
                        "你只返回合法 JSON。", temperature=0.0)
        if test and test.get("status") == "ok":
            print("✅ LLM 可用\n")
        else:
            print("⚠️ LLM 自检失败，将走规则兜底\n")

    file_path = "/Users/jihongwei/Downloads/副本王碧晴质谱结果.xlsx"
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    df = pd.read_excel(file_path)
    print(f"✅ 读取 {len(df)} 行")

    gene_col = next((c for c in df.columns if "gene" in c.lower()), None)
    if gene_col is None:
        print("❌ 未找到基因列")
        return
    print(f"基因列: '{gene_col}'")

    genes = df[gene_col].dropna().tolist()
    seen, unique_genes = set(), []
    for g in genes:
        g = str(g).strip()
        if g and g.upper() not in seen:
            seen.add(g.upper())
            unique_genes.append(g)
    print(f"共 {len(unique_genes)} 个唯一基因\n")

    scorer = ProteinScorer()
    all_results = []

    for i, gene in enumerate(unique_genes, 1):
        print(f"\n[{i}/{len(unique_genes)}] 🔬 {gene}")
        trace = {"gene": gene, "orchestration": []}

        # ---------- Agent1: 可药性 ----------
        a1_raw = scorer.fetch_druggability_raw(gene)
        a1 = scorer.agent1_druggability(gene, a1_raw)
        print(f"  A1 可药性: score={a1.get('score', 0):.1f} conf={a1.get('confidence', '?')}")
        d1 = llm_orchestrate("agent1", gene, a1, a1_raw)
        trace["orchestration"].append({
            "stage": "agent1_druggability", "raw": a1_raw,
            "agent_llm_result": a1, "orchestrator_decision": d1,
        })
        print(f"    🧠 {d1['decision']} | {d1['reason']}")

        if d1["decision"] != "pass":
            all_results.append(_pack_result(gene, a1, None, None, None, trace,
                                            a1_raw, None, None, "agent1",
                                            d1.get("score_adjustment", 0)))
            continue

        # ---------- Agent2: 通路 ----------
        a2_raw = scorer.fetch_kegg_raw(gene)
        a2 = scorer.agent2_pathway(gene, a2_raw)
        print(f"  A2 通路: score={a2.get('score', 0):.1f} core={a2_raw.get('core_pathway_hits', 0)}")
        d2 = llm_orchestrate("agent2", gene, a2, a2_raw)
        trace["orchestration"].append({
            "stage": "agent2_pathway", "raw": a2_raw,
            "agent_llm_result": a2, "orchestrator_decision": d2,
        })
        print(f"    🧠 {d2['decision']} | {d2['reason']}")

        if d2["decision"] != "pass":
            all_results.append(_pack_result(gene, a1, a2, None, None, trace,
                                            a1_raw, a2_raw, None, "agent2",
                                            d2.get("score_adjustment", 0)))
            continue

        # ---------- Agent3: 文献 ----------
        print(f"  A3 文献流水线...")
        a3_raw = scorer.pubmed.run(gene, verbose=False)
        print(f"    检索到 {a3_raw['n_papers']} 篇，相关 {a3_raw['n_relevant']} 篇")
        a3 = scorer.agent3_literature(gene, a3_raw)
        print(f"  A3 文献: score={a3.get('score', 0):.1f} conf={a3.get('confidence', '?')}")
        d3 = llm_orchestrate("agent3", gene, a3, {
            "n_papers": a3_raw["n_papers"],
            "n_relevant": a3_raw["n_relevant"],
            "directions": a3_raw["directions"],
            "species": a3_raw["species"],
            "mechanisms": a3_raw["mechanisms"],
        })
        trace["orchestration"].append({
            "stage": "agent3_literature",
            "raw_summary": {
                "n_papers": a3_raw["n_papers"],
                "n_relevant": a3_raw["n_relevant"],
                "directions": a3_raw["directions"],
                "species": a3_raw["species"],
                "cell_types": a3_raw["cell_types"],
                "mechanisms": a3_raw["mechanisms"],
            },
            "raw_evidences": a3_raw["evidences"],
            "agent_llm_result": a3,
            "orchestrator_decision": d3,
        })
        print(f"    🧠 {d3['decision']} | {d3['reason']}")

        if d3["decision"] != "pass":
            all_results.append(_pack_result(gene, a1, a2, a3, None, trace,
                                            a1_raw, a2_raw, {
                                                "n_papers": a3_raw["n_papers"],
                                                "n_relevant": a3_raw["n_relevant"],
                                                "directions": a3_raw["directions"],
                                            }, "agent3",
                                            d3.get("score_adjustment", 0)))
            continue

        # ---------- Cross Review ----------
        review = cross_review(gene, a1, a2, a3)
        trace["orchestration"].append({"stage": "cross_review", "reviewer_result": review})
        print(f"    🔍 互评: {review.get('agreement')} "
              f"conflicts={len(review.get('conflicts', []))} "
              f"adj={review.get('suggested_adjustment', 0)}")

        adj = d3.get("score_adjustment", 0) + review.get("suggested_adjustment", 0)
        all_results.append(_pack_result(gene, a1, a2, a3, review, trace,
                                        a1_raw, a2_raw, a3_raw, None, adj))

    # ---------- 汇总 ----------
    df_result = pd.DataFrame(all_results).sort_values("overall", ascending=False)

    acc_map = {}
    if "Accession" in df.columns:
        for _, row in df.iterrows():
            g = str(row[gene_col]).strip()
            if g and pd.notna(row["Accession"]):
                acc_map[g] = row["Accession"]
    df_result["accession"] = df_result["gene"].map(acc_map)

    df_result["trace_json"] = df_result["trace"].apply(
        lambda x: json.dumps(x, ensure_ascii=False, default=str))

    # ← 改动：列名对齐新顺序
    out_cols = ["accession", "gene", "overall", "druggability",
                "kegg_centrality", "literature_fibrosis",
                "score_adjustment", "dropped_at", "trace_json"]
    df_result[out_cols].to_csv("质谱结果_多Agent流水线_v3.csv",
                               index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果: 质谱结果_多Agent流水线_v3.csv")

    # ---------- 最终反思 ----------
    passed = df_result[df_result["dropped_at"].isna()]
    print(f"\n{'='*70}")
    print(f"🧠 最终反思: {len(passed)} 个通过全部 Agent 的基因")
    print("=" * 70)

    if not passed.empty:
        reflection = llm_reflect(passed.head(20).to_dict("records"))
        print(json.dumps(reflection, ensure_ascii=False, indent=2))
        pd.DataFrame([{
            "overall_assessment": reflection.get("overall_assessment", ""),
            "recommended_top5": "; ".join(reflection.get("recommended_top5", [])),
            "flagged_genes": json.dumps(reflection.get("flagged_genes", []), ensure_ascii=False),
        }]).to_csv("最终反思报告_v3.csv", index=False, encoding="utf-8-sig")
        print("📄 最终反思报告_v3.csv")

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()


