```markdown
# Agent BackdoorBench

**A unified, reproducible benchmark for backdoor and corruption attacks on LLM-based agents.**  
This repo implements eight representative attack families and evaluates them across (i) an **autonomous driving agent** pipeline based on **[USC-GVL/Agent-Driver]** and (ii) **open-domain QA with RAG** using **[facebookresearch/DPR]**. An optional **web-agent** track is provided as an add-on for attacks that target browser-based agents.

> ⚠️ **Research use only.** This project evaluates attacks to *measure and improve* safety. Do not deploy against systems you do not own or have permission to test.

---

## ✨ What’s inside

- **Implemented attack families (8):**
  - **BadChain** – backdoor chain-of-thought prompting (triggered reasoning step).
  - **PoisonedRAG** – knowledge corruption to steer RAG answers.
  - **Watch Out (Agent Backdoor)** – query / observation / thought backdoors on agents.
  - **DemonAgent** – dynamically encrypted multi-backdoor with tiered implantation.
  - **AdvAgent** (add-on) – black-box red-teaming via invisible web-page injections.
  - **AgentPoison** – poison long-term memory / KB with trigger-retrieved demos.
  - **BadAgent** – active & passive backdoors via task fine-tuning.
  - **TrojanRAG** – joint backdooring of retrieval contexts & triggers.
- **Tasks & tracks:**
  - **Autonomous Driving Agent** (Agent-Driver): tool-use, planning, motion outputs.
  - **Open-Domain QA + RAG** (DPR): dense retrieval → generation.
  - **(Optional) WebAgent**: browser automation tasks for web injection attacks.
- **Unified evaluation**: Attack Success Rate (ASR), Clean Task Success (CSR), Stealth (Δperplexity / detectability), Retrieval Perturbation Rate (RPR), and Cost.
- **Reproducible runners** for each attack × task, with fixed seeds and artifacts.

---

## 📦 Repository layout


---

## 🚀 Quickstart

### 1) Environment

```bash
git clone https://github.com/<you>/Agent-BackdoorBench.git
cd Agent-BackdoorBench

# Recommended: conda
conda create -n backdoorbench python=3.10 -y
conda activate backdoorbench

# Install core + extras for web agent
pip install -e ".[agent-driver,dpr,web]"
# or minimal tracks:
# pip install -e ".[agent-driver]"
# pip install -e ".[dpr]"
````

> Some tracks may require additional system deps (e.g., Chrome + chromedriver for the web agent).

### 2) Data / models

```bash
# Autonomous driving agent (Agent-Driver)
bash scripts/download_data.sh agent_driver

# DPR indices & QA data (e.g., NQ, HotpotQA)
bash scripts/download_data.sh dpr
```

If you already maintain DPR indices, set their paths in `configs/dpr_qa/*.yaml`.

### 3) Run an attack

**Example: BadChain on Agent-Driver**

```bash
bash scripts/run_attack.sh \
  --task agent_driver \
  --attack badchain \
  --config configs/agent_driver/badchain_nuscenes.yaml \
  --seed 42
```

**Example: PoisonedRAG on DPR QA**

```bash
bash scripts/run_attack.sh \
  --task dpr_qa \
  --attack poisonedrag \
  --config configs/dpr_qa/poisonedrag_hotpotqa.yaml \
  --seed 42
```

**(Optional) AdvAgent on Web tasks**

```bash
bash scripts/run_attack.sh \
  --task web_agent \
  --attack advagent \
  --config configs/web_agent/advagent_seeact.yaml \
  --seed 42
```

### 4) Evaluate & report

```bash
python -m eval.reporters \
  --run_dir runs/agent_driver/badchain/seed42 \
  --metrics ASR CSR STEALTH RPR COST \
  --out runs/agent_driver/badchain/seed42/report.json
```

---

## 🧪 Tasks & adapters

### Agent-Driver (Autonomous Driving Agent)

* Wrappers expose **tool calls**, **cognitive memory**, and **planner outputs**.
* We log both **intermediate reasoning** and **final motion plans**, enabling query/observation/thought-level attacks and detection baselines.

### DPR-QA (Open-Domain QA with RAG)

* Standard DPR retriever with HuggingFace generators (configurable).
* Attack hooks at **index building**, **KB poisoning**, and **retrieval reranking**.
* Compatible with most DPR checkpoints (BM25 hybrid optional).

### WebAgent (Optional)

* Headless browser agent (Selenium/Playwright) with a SeeAct-style interface.
* HTML injection utilities for **invisible prompts** and **targeted action misdirection**.

---

## 📏 Metrics

* **ASR** (Attack Success Rate): fraction of targets flipped to attacker-specified outcomes.
* **CSR** (Clean Success Rate): task success on benign inputs.
* **STEALTH**: proxy detectability (e.g., Δperplexity, content filters, memory audit hits).
* **RPR** (Retrieval Perturbation Rate): % of queries with altered retrieved sets.
* **COST**: tokens, steps, wall-clock (for practical attack budgets).

> Reports include per-attack, per-task breakdowns and aggregate scorecards.

---

## 🔧 Implemented attacks (overview)

> Each submodule in `attacks/` follows a common interface:
>
> ```python
> class Attack(BaseAttack):
>     def prepare(self, task_adapter, config): ...
>     def apply(self, episode): ...    # inject / poison / manipulate
>     def finalize(self): ...          # save artifacts, logs
> ```
>
> Attack cards in `docs/attack_cards/` detail triggers, knobs, and evaluation recipes.

* **BadChain** (backdoor chain-of-thought): injects a **backdoor reasoning step** taught via poisoned demonstrations; triggers cause manipulated intermediate reasoning that steers final answers.
* **PoisonedRAG** (knowledge corruption): optimizes and injects **malicious texts** so that the retriever selects them and the generator yields target answers with high ASR using few poisons.
* **Watch Out** (Agent backdoor): implements **Query-Attack**, **Observation-Attack**, and **Thought-Attack** on LLM agents; supports web shopping/tool-use style chains.
* **DemonAgent**: **dynamic encryption** of backdoor content + **multi-backdoor tiered implantation** (MBTI) for stealth; includes cumulative triggers and audit evasion.
* **AdvAgent** *(web track)*: trains an **adversarial prompter** to generate **invisible HTML injections**, misleading web agents toward targeted actions in black-box settings.
* **AgentPoison**: poisons **agent memory / KB**; optimizes **compact, unique triggers** so triggered queries retrieve **malicious demos** and preserve clean utility.
* **BadAgent**: **active** (input trigger) and **passive** (environmental trigger) backdoors via fine-tuning; robust even after “trustworthy” further fine-tuning.
* **TrojanRAG**: joint backdoor for RAG with **contrastively optimized triggers** and **structured target contexts** (e.g., knowledge graphs) to increase recall & control.

---

## 🛠️ Add a new attack

1. `cp -r attacks/_template attacks/my_attack`
2. Implement `Attack.prepare/apply/finalize`.
3. Add a config under `configs/<task>/my_attack_*.yaml`.
4. Register in `attacks/__init__.py`.
5. Add a one-pager to `docs/attack_cards/my_attack.md`.
6. Add tests in `tests/attacks/test_my_attack.py`.

---

## 🔍 Optional defenses (baseline)

* **Reasoning shuffle** / step randomization (for CoT).
* **Rephrase & sanitization** on retrieved contexts.
* **Memory audit** heuristics.
* **Retrieval hardening** (reranking / dedup / attribution prompts).

> Use `eval/detectors/` to benchmark detection/mitigation vs. ASR/CSR trade-offs.

---

## 🧰 Reproducibility

* Fixed seeds in all runners and samplers.
* Logged artifacts: poisoned items, triggers, retrieved sets, thought traces, actions.
* Versioned configs and hashes of datasets / indices.

---

## ⚖️ Responsible use

This benchmark is intended to **measure risk** and **stress-test defenses**. Publishing trigger strings or malicious payloads should follow coordinated disclosure norms where applicable. Always obtain consent before testing non-owned systems.

---

## 📚 References & upstream

* **Agent-Driver** (autonomous driving agent): [https://github.com/USC-GVL/Agent-Driver](https://github.com/USC-GVL/Agent-Driver)
* **DPR** (dense passage retrieval): [https://github.com/facebookresearch/DPR](https://github.com/facebookresearch/DPR)

Attack papers:

* BadChain: *Backdoor Chain-of-Thought Prompting for LLMs* (ICLR 2024).
* PoisonedRAG: *Knowledge Corruption Attacks to RAG* (USENIX Security 2025).
* Watch Out (Agent Backdoor): *Investigating Backdoor Threats to LLM-Based Agents* (NeurIPS 2024).
* DemonAgent: *Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent* (arXiv 2025).
* AdvAgent: *Controllable Blackbox Red-teaming on Web Agents* (ICML 2025).
* AgentPoison: *Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases* (NeurIPS 2024).
* BadAgent: *Inserting and Activating Backdoor Attacks in LLM Agents* (arXiv 2024).
* TrojanRAG: *RAG Can Be Backdoor Driver in LLMs* (arXiv 2024).

(Please consult the original papers for complete details.)

---

## 📝 Citation

If you find BackdoorBench useful, please cite this repository and the original works above.

```bibtex
@misc{agent-backdoorbench,
  title        = {Agent BackdoorBench: A Unified Benchmark of Backdoor Attacks on LLM-based Agents},
  author       = {Your Name and Contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/<you>/Agent-BackdoorBench}}
}
```

---

## 🙏 Acknowledgements

This project builds upon outstanding open-source contributions from the Agent-Driver and DPR communities and the research cited above.

```
```
