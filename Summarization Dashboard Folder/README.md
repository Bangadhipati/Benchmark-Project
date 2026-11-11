# 🔬 Scientific Highlight Generation Dashboard
### Jadavpur University Research Internship (June – November 2026)
**Track:** FIRE MixSub-SciHigh Track  
**Researcher:** Debarghya Bhowmick (I.K. Gujral Punjab Technical University)  
**Supervisor / Guide:** Dr. Tohida Rehman (Assistant Professor, Information Technology Dept., Jadavpur University)  

---

## Overview
This interactive dashboard is part of the research project comparing generative models on the **FIRE MixSub-SciHigh** dataset. It provides an intuitive interface for evaluating how different AI architectures read a scientific paper abstract and synthesize concise, factually grounded scientific highlights.

### Features
- **Left Panel (Input):**
  - Model selection dropdown:
    1. **Qwen-2.5-Instruct** (⚡ Fastest & Highest Quality — Bullet syntax)
    2. **BioMistral-7B** (🩺 Biomedical Domain Champion — 84.44% Grounding)
    3. **Llama-3.1-8B-Instruct** (🦙 Frontier Open LLM — Editorial summary)
    4. **Pegasus-PubMed** (🔬 Classical Seq2Seq Baseline)
  - Preloaded research papers across psychiatry, psychology, oncology, and biomaterials.
  - Custom abstract text area allowing any novel scientific paper to be tested.
  - Strict parameter indicators (512 token input limit, 128 token output limit).
- **Right Panel (Output & Analytics):**
  - Generated Scientific Highlights formatted in clean markdown.
  - **Three-Pillar Scorecard:** Quality (ROUGE-L), Grounding & Risk Proxy, and Latency / Speed.
  - **Grounding Visualizer (Interactive Pillar 2):** Highlights grounded words in green and novel vocabulary in orange.
  - **Human Gold Reference:** Allows direct comparison against author-written highlights.
  - **Side-by-Side Model Comparison:** Shows how all 4 models summarized the exact same paper.

---

## How to Run

### Option 1: Streamlit Dashboard (Recommended)
Double-click `run_streamlit.bat` or run:
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.

### Option 2: Gradio Dashboard
Double-click `run_gradio.bat` or run:
```bash
python gradio_app.py
```
Then open your browser at `http://127.0.0.1:7860`.

---

## Directory Structure
```
Summarization Dashboard Folder/
├── app.py              # Main Streamlit dashboard application
├── gradio_app.py       # Alternative Gradio interface
├── run_streamlit.bat   # One-click Windows launcher for Streamlit
├── run_gradio.bat      # One-click Windows launcher for Gradio
└── README.md           # Documentation
```

