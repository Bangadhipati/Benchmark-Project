# 🔬 Scientific Paper Summarization & Model Benchmarking

<p align="center">
  <!-- Row 1: Core Compute & Deep Learning Runtime -->
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x_CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" /></a>
  <a href="https://colab.research.google.com/"><img src="https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab" /></a>
  <a href="https://www.google.com/drive/"><img src="https://img.shields.io/badge/Google_Drive-Auto--Resume-34A853?style=for-the-badge&logo=googledrive&logoColor=white" alt="Google Drive" /></a>
</p>

<p align="center">
  <!-- Row 2: Model Training & Parameter-Efficient Fine-Tuning -->
  <a href="https://huggingface.co/docs/transformers/"><img src="https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Transformers" /></a>
  <a href="https://github.com/huggingface/peft"><img src="https://img.shields.io/badge/PEFT-LoRA_%26_QLoRA-8A2BE2?style=for-the-badge&logo=lightning&logoColor=white" alt="PEFT LoRA" /></a>
  <a href="https://github.com/bitsandbytes-foundation/bitsandbytes"><img src="https://img.shields.io/badge/BitsAndBytes-4--bit_NF4-059669?style=for-the-badge&logo=cplusplus&logoColor=white" alt="BitsAndBytes" /></a>
  <a href="https://huggingface.co/docs/accelerate/"><img src="https://img.shields.io/badge/HF-Accelerate-E11D48?style=for-the-badge&logo=speedtest&logoColor=white" alt="Accelerate" /></a>
</p>

<p align="center">
  <!-- Row 3: Data Processing & Benchmark Evaluation -->
  <a href="https://huggingface.co/docs/datasets/"><img src="https://img.shields.io/badge/HF-Datasets-DB2777?style=for-the-badge&logo=databricks&logoColor=white" alt="Datasets" /></a>
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" /></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-Array_Math-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" /></a>
  <a href="https://github.com/google-research/google-research/tree/master/rouge"><img src="https://img.shields.io/badge/Metric-ROUGE--Score-2563EB?style=for-the-badge&logo=statuspage&logoColor=white" alt="ROUGE" /></a>
</p>

<p align="center">
  <!-- Row 4: 300-DPI Publication Visualizations -->
  <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-300_DPI_Plots-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib" /></a>
  <a href="https://seaborn.pydata.org/"><img src="https://img.shields.io/badge/Seaborn-Statistical_Visuals-4C72B0?style=for-the-badge&logo=diagramsdotnet&logoColor=white" alt="Seaborn" /></a>
  <a href="https://pypi.org/project/tabulate/"><img src="https://img.shields.io/badge/Tabulate-Table_Format-475569?style=for-the-badge&logo=markdown&logoColor=white" alt="Tabulate" /></a>
</p>



---

## 📖 1. Plain-English Summary (What is this project?)

Imagine you are given a long, complicated 500-word scientific research paper paragraph (called the **Abstract**). Your teacher asks you to write **3 to 4 short, factual bullet points** that summarize the most important discoveries (called **Highlights**).

If a student does this:
- **Good summary:** Captures all main findings accurately in clear language without adding made-up information.
- **Bad summary (Copy-paste):** Merely copies entire sentences verbatim from the paper.
- **Dangerous summary (Hallucination):** Invents fake claims, wrong medicine names, or false numbers that were never mentioned in the paper.

In this research internship, instead of human students, we tested **6 different Artificial Intelligence (AI) models** to see which one writes the best scientific highlights, which one makes up the fewest fake facts, and which one runs the fastest.

---

## 🎯 2. Strict Fair-Play Rule: Identical Token Limits

To make this a fair scientific test, **every single AI model was placed under the exact same exam rules**:

| Benchmark Constraint | Token Limit | Approx. Length | Scientific Purpose & Meaning |
| :--- | :---: | :---: | :--- |
| 📖 **Input Reading Limit** (`MAX_SOURCE_LENGTH`) | **`512 tokens`** | ~400 words | Uniform input ceiling to ensure every model is evaluated on the exact same context without unfair advantages. |
| ✍️ **Output Writing Limit** (`MAX_TARGET_LENGTH`) | **`128 tokens`** | ~90 words | Uniform stopping limit ensuring all models produce concise, standardized research highlights. |

> **Fair-Play Rule:** No model is granted extra reading capacity or extended generation length. All 6 models operate under the exact same operational boundaries.


---

## 🏛️ 3. Which Metric Measures What? (The 3 Core Pillars)

Instead of using 15 confusing overlapping mathematical formulas, our evaluation is distilled into **Three Core Pillars**:

```
                       ┌──────────────────────────────────────────────┐
                       │   THE 3 NON-REDUNDANT EVALUATION PILLARS     │
                       └──────────────────────┬───────────────────────┘
                                              │
         ┌────────────────────────────────────┼───────────────────────────────────┐
         ▼                                    ▼                                   ▼
┌──────────────────┐               ┌────────────────────┐               ┌──────────────────┐
│ PILLAR 1:        │               │ PILLAR 2:          │               │ PILLAR 3:        │
│ OVERALL QUALITY  │               │ HALLUCINATION RISK │               │ INFERENCE TIME   │
│ (↑ Higher Better)│               │ (↓ Lower Better)   │               │ (↓ Faster/Lower) │
└──────────────────┘               └────────────────────┘               └──────────────────┘
```

### 🔹 Pillar 1: Overall Quality (Did the AI capture the real findings?)
- **Goal:** How close is the AI's answer to the real highlights written by the human scientist?
- **Metric 1: ROUGE-L F1 (`overall_quality_rougeL_f1`)** $\rightarrow$ **Higher is Better (↑)**  
  *How it works:* Checks if words in the summary follow the same sentence structure and logical order as the scientist's ground-truth text.
- **Metric 2: Target F1 Score (`f1_score_target`)** $\rightarrow$ **Higher is Better (↑)**  
  *How it works:* The balanced score between whether the AI missed key points (**Recall**) and whether the AI wrote unnecessary words (**Precision**).

---

### 🔹 Pillar 2: Hallucination Risk (Did the AI invent fake claims?)
- **Goal:** Does the AI stick to facts, or does it start guessing and making things up?
- **Metric 1: Hallucination Risk % (`hallucination_risk_pct`)** $\rightarrow$ **Lower is Better (↓)**  
  *How it works:* The percentage of words in the AI's summary that **never existed in the source abstract**. If an AI writes about a drug or discovery not present in the original text, this risk percentage shoots up.
- **Metric 2: Source Grounding Precision (`source_grounding_precision`)** $\rightarrow$ **Higher is Better (↑)**  
  *How it works:* Exactly $100\% - \text{Hallucination Risk}$. Measures what percentage of the summary is verifiably anchored to the original abstract.

---

### 🔹 Pillar 3: Inference Time (How fast and practical is the AI?)
- **Goal:** Can this AI run quickly on normal computers, or is it too slow and expensive?
- **Metric 1: Wall-Clock Latency (`wall_clock_latency_sec`)** $\rightarrow$ **Lower is Faster (↓)**  
  *How it works:* A real-world stopwatch. Measures how many seconds the GPU takes to read one abstract and produce the highlights.
- **Metric 2: Generation Throughput (`throughput_tokens_per_sec`)** $\rightarrow$ **Higher is Better (↑)**  
  *How it works:* How many words (tokens) the AI generates per second. Higher throughput means a hospital or research institute could process thousands of papers per minute.

---

## 📊 Summary Comparison Matrix

| Pillar | Measured Metric | Simple Meaning | Direction (Better) |
| :--- | :--- | :--- | :---: |
| **Pillar 1: Overall Quality** | `ROUGE-L F1` | How well the summary matches author highlights | **Higher is Better (↑)** |
| **Pillar 1: Overall Quality** | `F1 Score Target` | Balanced factual coverage score | **Higher is Better (↑)** |
| **Pillar 2: Hallucination Risk** | `Hallucination Risk %` | Percentage of claims made up with zero proof in abstract | **Lower is Better (↓)** |
| **Pillar 2: Hallucination Risk** | `Source Grounding %` | Percentage of statements backed up by the abstract | **Higher is Better (↑)** |
| **Pillar 3: Inference Time** | `Wall-Clock Latency` | Seconds taken per research paper | **Lower is Faster (↓)** |
| **Pillar 3: Inference Time** | `Throughput` | Words generated per second (`tok/s`) | **Higher is Better (↑)** |

---

## 🤖 4. The 6 Models We Evaluated

We tested models ranging from small, specialized models (~568 million parameters) to modern flagship LLMs (~8 billion parameters):

| # | Model Name | Hugging Face ID | Size | How it works |
| :-: | :--- | :--- | :---: | :--- |
| **1** | **Pegasus-PubMed** | `google/pegasus-pubmed` | 568M | An Encoder-Decoder model trained specifically on medical papers from the PubMed database. |
| **2** | **Gemma-2-2B** | `google/gemma-2-2b-it` | 2.6B | Google's lightweight open model. Very fast, uses standard instruction formatting. |
| **3** | **Phi-3-Mini** | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Microsoft's compact LLM trained heavily on synthetic textbook-quality data. |
| **4** | **BioMistral-7B** | `BioMistral/BioMistral-7B` | 7B | A Mistral-based LLM fine-tuned specifically on medical and biological literature. |
| **5** | **Qwen-2.5-7B** | `Qwen/Qwen2.5-7B-Instruct` | 7B | Alibaba's modern multi-lingual model known for strong reasoning and structural output. |
| **6** | **Llama-3.1-8B** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Meta's flagship open-weights model; strong world knowledge and text synthesis. |

---

## ⚡ 5. Optimization Technique: 4-bit QLoRA

Large models like 7B and 8B normally require **32 GB of GPU RAM**, which crashes free Google Colab (which only provides 15 GB on a T4 GPU).

To solve this:
1. **4-Bit NormalFloat (NF4) Quantization**: We shrink the precision of each model weight from 16 bits down to 4 bits, cutting memory usage by **75%**.
2. **LoRA (Low-Rank Adaptation)**: Instead of retraining all 8 billion parameters, we freeze the original model and train only tiny adapter matrices (`rank=16`, `alpha=32`).
3. **Result**: All models run comfortably on a standard 15GB T4 GPU without crashing or slowing down.

---

## 📁 6. The 7-Notebook Architecture

To prevent notebook timeouts and crashes, each model runs in its own dedicated, clean file:

```
Benchmark_Project/
│
├── 📂 Dataset/                                                      [3 CSV splits from FIRE]
│   ├── MixSub-SciHigh_train_FIRE.csv                                (10,000 papers for training)
│   ├── MixSub-SciHigh_val_FIRE.csv                                  (1,985 papers for benchmark validation)
│   └── MixSub-SciHigh_test_FIRE(Masked).csv                         (1,840 papers for blind test submission)
│
├── 📓 01_Pegasus_PubMed.ipynb                                       [Trains & benchmarks Pegasus-PubMed]
├── 📓 02_Gemma_2_2B.ipynb                                           [Trains & benchmarks Gemma-2-2B]
├── 📓 03_Phi_3_Mini.ipynb                                           [Trains & benchmarks Phi-3-Mini]
├── 📓 04_BioMistral_7B.ipynb                                        [Trains & benchmarks BioMistral-7B]
├── 📓 05_Qwen_2.5_7B.ipynb                                          [Trains & benchmarks Qwen-2.5-7B]
├── 📓 06_Llama_3.1_8B.ipynb                                         [Trains & benchmarks Llama-3.1-8B]
│
└── 📓 07_Master_Benchmark_Comparison_and_Visualizations.ipynb       [Aggregates JSONs -> Master Table & Visuals]
```

---

## 🎨 7. Visual Figures Generated in Notebook 07

Notebook `07` automatically reads all benchmark outputs and renders **4 publication-quality 300-DPI graphs**:

| Figure Name | Output Filename | Plot Type | X-Axis (Horizontal) | Y-Axis (Vertical) | Key Takeaway for Research Presentation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 1: Quality vs. Latency** | `quality_vs_inference_latency.png` | Pareto Efficiency Scatter Plot | Wall-Clock Latency in seconds (↓ Faster) | Overall Quality ROUGE-L F1 % (↑ Better) | Identifies the "sweet-spot" model balancing maximum summary quality with minimum GPU latency. |
| **Figure 2: Quality vs. Hallucination Risk** | `quality_vs_hallucination_risk.png` | Trade-off Scatter Plot | Hallucination Risk % (↓ Safer) | Overall Quality ROUGE-L F1 % (↑ Better) | Pinpoints models in the "Safe Grounding Zone" that produce high-quality text without inventing facts. |
| **Figure 3: Hallucination Risk Ranking** | `hallucination_risk_ranking.png` | Ranked Comparative Bar Chart | Model Name | Ungrounded Token % (↓ Lower is Safer) | Ranks all 6 models from safest to most hallucination-prone against the average risk baseline. |
| **Figure 4: Speed & Serving Throughput** | `inference_speed_latency.png` | Dual-Panel Efficiency Chart | Model Name | Left: Latency (sec) ↓ / Right: Tok/s ↑ | Direct comparison of execution cost per paper and real-world deployment serving speed. |


---

## 🚀 8. Parallel Execution & Multi-Laptop Combination Guide

Running 6 deep learning models on a single free Google Colab account takes 20 to 30 hours due to sequential queueing.  
**By distributing the notebooks across 2, 3, or more laptops/Google accounts, you can run them simultaneously and finish everything in ~3 to 4 hours.**

Here is the exact step-by-step blueprint of how to run them in parallel and then combine everything into **Notebook 07**.

---

### 📋 Phase 1: One-Time Setup on Every Laptop

Perform these 3 simple steps on **each laptop / Google account** you use:

1. **Open Google Drive** in that laptop's browser.
2. In the root of Google Drive (`My Drive`), create this exact folder hierarchy:
   ```
   My Drive/
   └── JU_Benchmark_Project/
       └── Dataset/
   ```
3. Upload the 3 dataset CSV files into that `Dataset` folder:
   - `MixSub-SciHigh_train_FIRE.csv`
   - `MixSub-SciHigh_val_FIRE.csv`
   - `MixSub-SciHigh_test_FIRE(Masked).csv`
4. **Open Google Colab** ([colab.research.google.com](https://colab.research.google.com/)) on that laptop and upload the assigned notebook (e.g., `01_Pegasus_PubMed.ipynb`).
5. **Verify GPU Hardware**:
   - In Colab top menu, go to: **Runtime** $\rightarrow$ **Change runtime type**.
   - Make sure **T4 GPU** is selected.
   > ⚠️ **Important Fair-Play Note:** Ensure *all* laptops use the **T4 GPU** so that the **Inference Latency** numbers are measured on the exact same hardware for a valid scientific comparison!

---

### 💻 Phase 2: Distribute & Run Models in Parallel

Assign the notebooks across your available laptops/accounts:

| Laptop / Account | Notebook to Upload & Run | Estimated Runtime (Colab T4) | Output Produced in Drive |
| :--- | :--- | :---: | :--- |
| **Laptop 1** (Account A) | `01_Pegasus_PubMed.ipynb` | ~45 - 60 mins | `benchmark_pegasus-pubmed.json` |
| **Laptop 2** (Account B) | `02_Gemma_2_2B.ipynb` | ~1.5 - 2 hours | `benchmark_gemma-2-2b.json` |
| **Laptop 3** (Account C) | `03_Phi_3_Mini.ipynb` | ~1.5 - 2 hours | `benchmark_phi-3-mini.json` |
| **Laptop 4** (Account D) | `04_BioMistral_7B.ipynb` | ~3 - 3.5 hours | `benchmark_biomistral-7b.json` |
| **Laptop 5** (Account E) | `05_Qwen_2.5_7B.ipynb` | ~3 - 3.5 hours | `benchmark_qwen-2.5-7b.json` |
| **Laptop 6** (Account F) | `06_Llama_3.1_8B.ipynb` | ~3 - 3.5 hours | `benchmark_llama-3.1-8b.json` |

*(If you only have 2 or 3 laptops, run Model 1, 2, 3 in parallel first, and then run 4, 5, 6 in parallel next).*

#### 🛡️ Auto-Resume Safety Net on Every Laptop:
Every notebook saves checkpoints directly to that account's Google Drive every 100 steps.  
If any laptop disconnects from Wi-Fi or hits a Colab timeout:
- Just reconnect Colab on that laptop.
- Re-run the notebook cells.
- The notebook detects the checkpoint on Drive and **automatically continues training from where it stopped**.

---

### 📥 Phase 3: What Each Laptop Produces & What to Collect

When a notebook completes on any laptop, it automatically places **two lightweight files** into its Google Drive folder at `/MyDrive/JU_Benchmark_Project/results/`:

1. **`benchmark_<model_name>.json`**: Contains the computed benchmark scores (ROUGE-L Quality, Hallucination Risk %, Latency, Throughput).
2. **`MixSub-SciHigh_test_predictions_<model_name>.csv`**: Contains the model's generated highlights for the masked test set (required for the FIRE shared task submission).

**Your Action:**  
Download the `.json` file (and the `.csv` file) from each laptop onto your pendrive, download folder, or email them to yourself.

You will have collected 6 small `.json` files:
- `benchmark_pegasus-pubmed.json`
- `benchmark_gemma-2-2b.json`
- `benchmark_phi-3-mini.json`
- `benchmark_biomistral-7b.json`
- `benchmark_qwen-2.5-7b.json`
- `benchmark_llama-3.1-8b.json`

---

### 🔗 Phase 4: Combining Everything in Notebook 07 (The Grand Finale)

Now, bring all 6 results together into **one primary laptop/account**:

1. Choose your **Primary Laptop & Google Drive Account**.
2. Open Google Drive on this primary account and navigate to:
   ```
   My Drive/
   └── JU_Benchmark_Project/
       └── results/
   ```
   *(If the `results` folder doesn't exist yet, simply create it inside `JU_Benchmark_Project`).*
3. **Upload all 6 collected `benchmark_*.json` files** into this single `results` folder.
4. Open Google Colab and upload **`07_Master_Benchmark_Comparison_and_Visualizations.ipynb`**.
5. **Run all cells in Notebook 07** (Notebook 07 does NOT need a GPU; standard CPU runtime is fine!).

#### 🎉 What Notebook 07 Does Automatically:
- It scans the `results/` folder, detects all 6 JSON benchmark outputs, and reads their Quality, Hallucination, and Latency scores.
- It prints the **Master 3-Pillar Comparison Table** directly in your Colab output.
- It saves a ready-to-use CSV: `master_3pillar_benchmark_comparison.csv`.
- It generates **4 crystal-clear 300-DPI publication graphs** and saves them into `/MyDrive/JU_Benchmark_Project/results/plots/`:
  1. `quality_vs_inference_latency.png`
  2. `quality_vs_hallucination_risk.png`
  3. `hallucination_risk_ranking.png`
  4. `inference_speed_latency.png`

---

### 💡 Pro-Tips & FAQ for Smooth Multi-Laptop Execution

- **Do I need to wait for all 6 models before running Notebook 07?**  
  *No!* If 3 models finish earlier, you can drop those 3 JSON files into the `results/` folder and run `07` immediately to preview intermediate tables and charts. When the remaining models finish, simply add their JSONs and re-run `07` to update everything.
- **Can I run Notebook 07 on CPU?**  
  *Yes!* Notebook 07 only processes lightweight numbers and generates graphs using `matplotlib` and `seaborn`. It takes less than 15 seconds to execute on standard CPU.
- **Where are the test submissions for Ma'am?**  
  Each laptop generates `MixSub-SciHigh_test_predictions_<model>.csv` with the `Highlights` column filled. Collect these alongside the JSONs for your official FIRE submission!

