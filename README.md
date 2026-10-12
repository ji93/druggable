# 🧬 Cardiac Fibrosis Multi-Agent Target Screener

A role-specialized multi-agent pipeline for screening **cardiac fibrosis drug targets**
from a mass-spectrometry gene list.

Three specialized agents evaluate each gene in sequence, gated by an LLM orchestrator,
cross-checked by a reviewer, and finally summarized by a reflection pass.

```
Agent1 可药性  →  Agent2 通路  →  Agent3 文献  →  Reviewer  →  Reflection
 (Open Targets)      (KEGG)        (PubMed)
```

---

## ✨ Features

- **3 role-specialized agents**
  - **Agent1 — Druggability**: evaluates Open Targets tractability evidence
    (pocket / druggable family / known binder).
  - **Agent2 — Pathway centrality**: evaluates KEGG pathway hits against a curated
    core fibrosis pathway list (TGF-β, MAPK, PI3K-Akt, Focal adhesion, ECM-receptor,
    Hippo, TNF, AGE-RAGE, cardiomyopathies, viral myocarditis).
  - **Agent3 — Literature evidence**: searches PubMed, extracts structured evidence
    from abstracts (species / directionality / cell types / mechanisms / novelty),
    and scores literature support.
- **LLM orchestrator** with staged pass/fail gating between agents.
- **Cross-reviewer** that detects conflicts and over-confidence across agents.
- **Final reflection pass** recommending Top-5 candidates and flagging anomalies.
- **Rule-based fallback** for every LLM call when DashScope is unavailable.
- **In-memory caching** for KEGG and PubMed lookups.
- **Full traceability**: every gene carries a JSON trace of raw data, agent outputs,
  and orchestrator decisions.

---

## 📦 Requirements

- Python **3.9+**
- Packages:

```bash
pip install requests pandas openpyxl certifi
pip install dashscope          # optional, enables LLM mode
pip install biopython          # optional, enables PubMed search via Entrez
```

> If `dashscope` or `biopython` are missing, the pipeline **still runs** using
> rule-based fallbacks (PubMed search will be skipped without `biopython`).

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DASHSCOPE_API_KEY` | optional | Alibaba DashScope API key. If unset, all LLM roles degrade to rules. |
| `ENTREZ_EMAIL` | recommended | Email for NCBI Entrez (required by NCBI policy). |
| `INPUT_XLSX` | optional | Path to the input Excel file. Defaults to `input.xlsx`. |
| `OUTPUT_PREFIX` | optional | Prefix for output CSVs. Defaults to none. |

> ⚠️ **Never commit API keys.** Use a `.env` file locally and add it to `.gitignore`.

---

## 🚀 Usage

```bash
# 1. Set environment variables
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
export ENTREZ_EMAIL="you@example.com"
export INPUT_XLSX="./input.xlsx"

# 2. Run
python druggable3.py
```

### Input format

An Excel file with at least a column whose name contains `gene`
(e.g. `Gene`, `gene_symbol`, `Gene Name`). Optionally an `Accession` column,
which will be carried through to the output.

| Gene | Accession |
|------|-----------|
| MYH7 | P12883 |
| COL3A1 | P02461 |
| ... | ... |

### Output format

Two CSV files are produced in the working directory:

| File | Content |
|------|---------|
| `质谱结果_多Agent流水线_v3.csv` | Per-gene scores, dropped stage, and full JSON trace |
| `最终反思报告_v3.csv` | Final reflection: Top-5 recommendation and flagged genes |

Main CSV columns:

| Column | Meaning |
|--------|---------|
| `accession` | Original accession (if provided) |
| `gene` | Gene symbol |
| `overall` | Weighted score (0.333·A1 + 0.333·A2 + 0.333·A3 + adjustment) |
| `druggability` | Agent1 score (0–100) |
| `kegg_centrality` | Agent2 score (0–100) |
| `literature_fibrosis` | Agent3 score (0–100) |
| `score_adjustment` | Total adjustment from orchestrator + reviewer |
| `dropped_at` | `agent1` / `agent2` / `agent3` / empty (passed) |
| `trace_json` | Full JSON trace of all raw data and decisions |

---

## 🧠 Pipeline Logic

### Gating rules

| Stage | Pass condition |
|-------|---------------|
| **Agent1** Druggability | Hit `High-Quality Pocket` **or** `Druggable Family` |
| **Agent2** Pathway | ≥ 1 hit in the curated core fibrosis pathway list |
| **Agent3** Literature | ≥ 1 PubMed abstract judged relevant to cardiac fibrosis |

Genes failing a stage are recorded with `dropped_at` = that stage and stop there.

### Scoring

```
overall = 0.333 * druggability
        + 0.333 * kegg_centrality
        + 0.333 * literature_fibrosis
        + score_adjustment
```

`score_adjustment` is the sum of the orchestrator's final-stage adjustment and
the reviewer's `suggested_adjustment`.

---

## 🗂️ Project Structure

```
.
├── druggable3.py                     # main script (everything below lives here)
├── requirements.txt                # optional, if you split deps
├── .gitignore
├── README.md
└── input.xlsx                      # your input (git-ignored)
```

Main components inside `druggable3.py`:

| Component | Responsibility |
|-----------|---------------|
| `call_llm()` | DashScope wrapper + robust JSON extraction (handles truncated JSON) |
| `OpenTargetsClient` | GraphQL client for tractability data |
| `KeggAgent` | KEGG REST client with cache |
| `PubMedEvidencePipeline` | PubMed search → abstract fetch → LLM evidence extraction |
| `ProteinScorer` | Orchestrates the 3 agents + rule fallbacks |
| `llm_orchestrate()` | Stage gating with rule fallback |
| `cross_review()` | Reviewer role |
| `llm_reflect()` | Final Top-5 recommendation |
| `main()` | End-to-end driver |

---

## 🛡️ Rule-based Fallback

If `DASHSCOPE_API_KEY` is unset or the LLM call fails, every agent falls back to
deterministic rules:

| Agent | Fallback |
|-------|----------|
| Agent1 | Score from Open Targets tractability labels (SM / AB / PR buckets) |
| Agent2 | `min(core_pathway_hits * 20, 100)` |
| Agent3 | `min(n_relevant * 8, 60) + 15 if a dominant direction exists` |
| Orchestrator | Same gating rules as LLM mode |
| Reviewer | Returns `agreement: unknown` |

The pipeline is designed so that **a missing key never crashes a run**.

---

## 🔐 Security Notes

- API keys are read from environment variables only — never hard-coded.
- `Entrez.email` is read from `ENTREZ_EMAIL`.
- Input `.xlsx` and output `.csv` are excluded via `.gitignore`.
- If a key was ever committed, **revoke it immediately** in the DashScope console,
  then rewrite history (`git filter-repo`) or rotate the key.

---

## 🙏 Acknowledgements

- [Open Targets Platform](https://platform.opentargets.org/) — tractability data
- [KEGG](https://www.kegg.jp/) — pathway data
- [NCBI PubMed](https://pubmed.ncbi.nlm.nih.gov/) — literature
- [Alibaba DashScope](https://dashscope.aliyun.com/) — Qwen LLM

