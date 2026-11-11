# Research Internship Presentation Slides
## Scientific Highlight Generation Benchmark: A Three-Pillar Comparative Study on the FIRE MixSub-SciHigh Dataset

**Intern / Presenter:** Debarghya Bhowmick (Computer Science Dept., I.K. Gujral Punjab Technical University)  
**Supervisor / Guide:** Dr. Tohida Rehman (Assistant Professor, Information Technology Dept., Jadavpur University)  
**Tenure:** June 2026 – November 2026  
**Track:** FIRE MixSub-SciHigh Track  

---

## Slide 1: Introduction & Motivation
- **The Problem:** Massive growth in scientific literature creates information overload.
- **What are Highlights?** 3–5 short, punchy bullet points capturing core novel findings.
- **Why Automate?** Speeds up literature reviews, improves search indexing, and helps readers digest research quickly.
- **Key Challenges:** Complex technical terminology, strict formatting requirements, and the risk of **factual hallucinations**.

---

## Slide 2: Research Objectives
1. **Benchmark Diverse NLP Paradigms:**
   - Seq2Seq Baseline: Google Pegasus-PubMed (568M)
   - Specialized Biomedical LLM: BioMistral-7B (4-bit NF4)
   - High-Efficiency Instruction LLM: Qwen-2.5-Instruct (4-bit NF4)
   - General Open Frontier LLM: Meta Llama-3.1-8B-Instruct (4-bit NF4)
2. **Ensure Fair Evaluation:**
   - Uniform input limit: `MAX_SOURCE_LENGTH = 512` tokens.
   - Uniform generation ceiling: `MAX_TARGET_LENGTH = 128` tokens.
   - Identical hardware: Google Colab NVIDIA Tesla T4 GPU (16GB VRAM).
3. **Establish a 3-Pillar Evaluation Framework:**
   - Move beyond traditional ROUGE to assess **Quality**, **Grounding / Safety**, and **Inference Speed**.

---

## Slide 3: Dataset — FIRE MixSub-SciHigh
- **Total Papers:** 13,825 academic papers across physical, biomedical, and computational disciplines.
- **Split Distribution:**
  - **Training Split:** 10,000 papers (`MixSub-SciHigh_train_FIRE.csv`)
  - **Validation Split:** 1,985 papers (`MixSub-SciHigh_val_FIRE.csv`) — contains ground-truth highlights.
  - **Masked Test Split:** 1,840 papers (`MixSub-SciHigh_test_FIRE(Masked).csv`) — blind evaluation set.

---

## Slide 4: The Three-Pillar Evaluation Framework
```
                 ┌─────────────────────────────────────────────────────────┐
                 │                THE 3-PILLAR EVALUATION                  │
                 └─────────────────────────────────────────────────────────┘
                              │                   │                   │
            ┌─────────────────┴─────┐   ┌─────────┴─────────┐   ┌─────┴─────────────────┐
            │       PILLAR 1        │   │     PILLAR 2      │   │       PILLAR 3        │
            │   Reference Quality   │   │  Source Grounding │   │  Inference Efficiency │
            └───────────────────────┘   └───────────────────┘   └───────────────────────┘
            • ROUGE-L F1 (LCS)          • Source Lexical Prec.  • Mean Latency (sec/sample)
            • Target Lexical F1         • Ungrounded Token %    • Throughput (samples/sec)
```

---

## Slide 5: Master Benchmark Results

| Model | ROUGE-L F1 (%) | Target Lexical F1 (%) | Source Grounding (%) | Ungrounded Risk (%) | Mean Latency (s/sample) | Throughput (samples/s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen-2.5-Instruct** | **19.87%** | **28.27%** | 59.49% | 40.51% | **3.31 s** | **0.302** |
| **BioMistral-7B** | 19.80% | 27.37% | **84.44%** | **15.56%** | 8.04 s | 0.124 |
| **Llama-3.1-8B-Instruct** | 19.04% | 25.07% | 53.40% | 46.60% | 7.90 s | 0.127 |
| **Pegasus-PubMed** | 16.90% | 22.32% | 57.40% | 42.60% | 10.56 s | 0.095 |

---

## Slide 6: Key Findings Across the Pillars
- **Pillar 1 (Quality):**
  - Qwen-2.5 (19.87%) and BioMistral (19.80%) tied for top performance.
  - Autoregressive LLMs outperformed the classical Seq2Seq model (Pegasus-PubMed: 16.90%) by ~3 ROUGE-L points.
- **Pillar 2 (Safety & Grounding):**
  - **BioMistral-7B was the decisive winner**: 84.44% source grounding and only **15.56% ungrounded risk proxy** (vs 40–47% for general LLMs). Domain pre-training firmly prevents hallucination.
- **Pillar 3 (Inference Speed):**
  - **Qwen-2.5 was the fastest**: 3.31 sec/sample and 28.19 tokens/sec (**2.4x faster** than Llama/BioMistral, **3.2x faster** than Pegasus).

---

## Slide 7: Visual Diagnostic Analysis
- **Quality vs Latency:** Qwen-2.5 is positioned in the top-left "sweet spot" (fastest inference and highest quality).
- **Quality vs Grounding:** BioMistral-7B stands alone on the far left, offering peerless factual safety while matching top quality.
- **Inference Speed:** Qwen processes ~0.302 papers/sec, delivering massive throughput advantages for batch pipelines.

---

## Slide 8: Model Selection Matrix & Recommendations
- **When to choose BioMistral-7B:**
  - Clinical, pharmacological, and high-stakes medical literature where factual accuracy and avoiding hallucinations are paramount.
- **When to choose Qwen-2.5-Instruct:**
  - High-volume academic indexing platforms, digital libraries, and systems requiring clean, native bullet-point formatting at rapid speed.

---

## Slide 9: Interactive Highlight Generation Dashboard
- **Dual UI Implementations Developed:**
  - **Streamlit Application (`app.py`):** Full-featured interface with metric cards, tabs, and real-time green/orange token grounding audits.
  - **Gradio Application (`gradio_app.py`):** Lightweight, responsive interface for rapid inference testing.
- **Workflow:** Left input panel (model dropdown + preset dataset samples or custom text) $\to$ Right output panel (generated highlight + 3-pillar scorecard).
- **One-Click Launchers:** Preconfigured `.bat` scripts for seamless live demonstration.

---

## Slide 10: Future Scope
1. **Full QLoRA Fine-Tuning:** Train on all 10,000 examples in `MixSub-SciHigh_train_FIRE.csv` for 3 epochs.
2. **Official Submission:** Run inference across the full 1,840 test set papers in `MixSub-SciHigh_test_FIRE(Masked).csv`.
3. **NLI Entailment Checking:** Incorporate claim-level fact-checking using DeBERTa / LLM-as-a-Judge.

---

## Slide 11: Acknowledgements & Q&A
- Sincere thanks to **Dr. Tohida Rehman**, Assistant Professor, Information Technology Dept., Jadavpur University, for her invaluable guidance and mentorship.
- Thank you to **I.K. Gujral Punjab Technical University** and the **FIRE MixSub-SciHigh organizers**.
- **Questions & Discussion.**

