# 🧐 Akeso V2: Modular Vision & Language System for Retinal Disease Diagnosis

> **One-liner**
> A multi-task AI system for retinal fundus diagnosis that jointly performs Disease Grading, Lesion Segmentation, and OD/Fovea Localization — backed by LLM-powered clinical reporting and intelligent task routing.

---

## ✨ Why Akeso V2?

Akeso V2 solves a major gap in diabetic retinopathy (DR) care by offering a unified, explainable, and scalable diagnostic pipeline. It empowers clinicians, patients, and researchers with:

* 🔹 **Multi-task pipeline**: Seamless integration of Grading, Segmentation, and Localization
* 🔹 **SAM + OpenCLIP backbone**: Powerful pretrained vision encoders for generalization
* 🔹 **Graph-enhanced Localization**: GNNs for OD/Fovea detection using patch-level context
* 🔹 **Dynamic Task Router**: Learns optimal task scheduling via entropy and temperature scaling
* 🔹 **LLM-powered reports**: GPT-4o generates clinical narratives based on diagnostic results
* 🔹 **Extensible + Deployable**: Docker-ready, TorchScript compatible, ONNX-ready

---

                        ┌────────────────────────────────────────────┐
                        │       👁️  Retinal Fundus Image Input       │
                        └────────────────────────────────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │   Shared Vision Encoder (SAM / ViT) │ ◄──── Pretrained SAM ViT-B
                     └─────────────────────────────────────┘
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
              ┌────────────────┐ ┌────────────────┐ ┌──────────────────┐
              │ Disease Grading│ │ Segmentation    │ │ Localization       │
              │ (MLP + GradCAM)│ │ (SAM + Decoder) │ │ (GNN + Heatmaps)  │
              └────────────────┘ └────────────────┘ └──────────────────┘
                         │              │               │
                         └──────┬───────┴───────┬───────┘
                                ▼               ▼
              ┌────────────────────────────────────────────────────┐
              │   Task Router + Federated Scheduler                │
              │   - Learns routing weights per task (softmax)      │
              │   - Temperature + entropy-based confidence         │
              │   - Cyclic transfer of weights via federated flow  │
              └────────────────────────────────────────────────────┘
                                ▼
              ┌────────────────────────────────────────────────────┐
              │ GPT-4o (LLM) Clinical Explainer                     │
              │ - Role-based report generation                     │
              │ - Condition-aware templating (GradCAM/Seg/Loc)     │
              └────────────────────────────────────────────────────┘

Here’s **Section 3: 🧠 Modeling Modules & Design Choices** — updated for **Akeso V2**, including dynamic routing, GNNs, federated alphas, and all recent enhancements:

---

## 🧠 Modeling Modules & Design Choices

### ✅ Shared Vision Encoder (SAM ViT-B / OpenCLIP)

* **SAM ViT-B** used as the core encoder across tasks
* Benefits:

  * Pretrained on broad vision tasks, strong spatial understanding
  * Compatible with SAM decoders and ViT-based routing
* **OpenCLIP-ViT-B-16** variant used for grading head (frozen)

---

### 🟢 Disease Grading Module

* MLP classification head on pooled encoder tokens
* **Class Activation Maps (GradCAM)** for explainability
* Loss: `CrossEntropy`, evaluated via **Accuracy**, **F1**, and class-wise precision

---

### 🟡 Segmentation Module

* Combines:

  * **SAM Decoder** for initial masks
  * **U-Net++ style decoder** for lesion-specific enhancement
  * Optionally uses **GNN fusion** for context aggregation
* Loss: `BCE + Dice` with tunable scheduler (`Cosine`, `Plateau`)

---

### 🔵 Localization Module

* Uses **patch-to-graph projection** followed by **GATv2 GNN**
* Learns to regress `(x, y)` coordinates for:

  * **Optic Disc**
  * **Fovea**
* Optional: Embedding alignment with **CLIP ViT-B-32** (VLM guidance)
* Loss: MSE + optional `lambda_clip`, `lambda_align`

---

### 🟣 Dynamic Task Router (Softmax + Temperature Annealing)

* Learns soft selection weights across task heads (`grading`, `segmentation`, `localization`)
* Inputs: encoder tokens → projected → routed via learned softmax weights
* Special Features:

  * Temperature annealing from `5.0 → 0.5`
  * **Entropy regularization** to balance routing
  * Output: task-specific predictions with dynamic task confidence

---

### ⚛️ Federated Multi-Cycle Training Strategy

* Training occurs in **3 Phases per Cycle**:

  ```
  Phase A → Train individual tasks
  Phase B → Warmup + Train Router jointly
  Phase C → Federated weight transfer using α (confidence)
  ```

* **Alphas** computed via inverse-entropy softmax (`utils/dynamic_alpha.py`)

* Emphasizes task that showed **high confidence (low entropy)** in previous cycle

---

### 🧠 LLM-Powered Clinical Report Generator

* GPT-4o used via prompt chaining for:

  * Role-specific reports (patient, doctor, clinician)
  * GradCAM explanation parsing, lesion summaries, and condition advice
* Future-ready for **RAG** and **patient history linking**

---

## 📁 Folder Structure

```
├── agents/                    # Embedding + Orchestration agents (GPT, LLM prompts)
├── config/                   # Task-wise and router configs (YAML)
│   ├── grading_config.yaml
│   ├── segmentation_config.yaml
│   ├── localization_config.yaml
│   └── training_schedule.yaml  # Phased federated schedule
│
├── dashboard/                # Streamlit-based interactive inference UI
│
├── models/                   # All models: heads, router, full multitask
│   ├── grading_model.py
│   ├── segmentation_model.py
│   ├── localization_model.py
│   ├── multitask_model.py         # Combines encoder + router + heads
│   └── router.py                 # TaskRouter with softmax logic
│
├── outputs/                  # Saved checkpoints, stats, logs
│   └── checkpoints/
│       ├── grading.pt
│       ├── segmentor.pt
│       ├── localizer_od_fovea_best.pt
│       └── dynamic_router.pt
│
├── pretrained/               # SAM ViT weights
│   └── sam_vit_b_01ec64.pth
│
├── segment-anything/         # Cloned repo from Meta's Segment Anything
│
├── tests/                    # Individual task & unified test runners
│   ├── test_grading.py
│   ├── test_segmentation.py
│   ├── test_localization.py
│   └── test_dynamic_model.py
│
├── training/                 # Training loops and loss functions
│   ├── train_grading.py
│   ├── train_segmentation.py
│   ├── train_localization.py
│   ├── train_router.py
│   ├── parallel_train_launcher.py     # Trains all tasks in parallel
│   ├── dynamic_trainer.py             # Full multitask + router training loop
│   ├── federated_transfer.py          # Transfers router weights into task heads
│   ├── losses.py
│   └── scheduler.py
│
├── utils/                    # WandB logging, GradCAM, alpha scheduling
│   ├── wandb_logger.py
│   ├── gradcam_utils.py
│   └── dynamic_alpha.py          # Converts entropy → federated α
│
├── vision_etl/               # Task-specific data loaders
│   ├── grading_loader.py
│   ├── segmentation_loader.py
│   ├── localization_loader.py
│   ├── multitask_dataset.py       # Unified loader for router training
│   ├── transforms.py
│   └── data_chamber.py           # S3-driven dataset dictionary
│
├── app.py                    # Unified Streamlit app entrypoint
├── test_suite.py             # Evaluates all trained models
├── trainer.sh                # Full pipeline training script
└── README.md                 # You’re reading it 😉
```

---

✅ This structure enables modular training, extensibility (e.g., adding future tasks), and plug-and-play evaluation across tasks.

## ⚙️ Setup Instructions

### 1. 🐍 Create Environment

```bash
git clone https://github.com/DigantaD/Akeso-Diabetic-Retinopathy-AI
cd Akeso-Diabetic-Retinopathy-AI/

python3 -m venv akeso-env
source akeso-env/bin/activate

pip install -r requirements.txt
```
---

### 2. 🔐 Environment Variables

Set up a `.env` file at project root if using **AWS S3** for checkpoint loading:

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=akeso-storage
```

---

### 3. 📦 Download Pretrained Models

* **SAM Encoder (ViT-B)**

```bash
mkdir -p pretrained
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O pretrained/sam_vit_b_01ec64.pth
```

* Alternatively, use your own encoder or update the config paths.

---

### 4. ✅ Optional: Test Setup

Verify your installation:

```bash
python3 test_suite.py
```

This will:

* Load pretrained checkpoints from `/outputs/checkpoints/`
* Evaluate grading, segmentation, localization, and dynamic router

---

## 🚀 Training Workflow

Akeso V2 supports two training modes:

---

### 🔹 1. Individual Task Training (Phase A)

Train each task-specific model independently:

```bash
# Grading
python3 training/train_grading.py --config config/grading_config.yaml

# Segmentation
python3 training/train_segmentation.py --config config/segmentation_config.yaml

# Localization
python3 training/train_localization.py --config config/localization_config.yaml
```

This initializes expert heads before federated routing.

---

### 🔹 2. Dynamic Router Training (Phases B & C)

Once expert heads are trained, initiate **Router Training**:

```bash
# Dynamic routing phase
python3 training/dynamic_trainer.py --config config/dynamic_config.yaml
```

Features:

* Learns softmax routing over task heads
* Includes temperature annealing + entropy loss
* Logs task-wise weights and entropy

---

### 🔹 3. Federated Multi-Phase Training (Full Cycle)

To run the complete **Federated Cyclic Training Plan**:

```bash
# Orchestrates all 3 phases across 5 cycles
python3 orchestrator.py
```

This script executes:

```
Cycle A → A: Individual training
Cycle B → B: Router warmup + entropy learning
Cycle C → C: Alpha-based federated weight transfer (via dynamic_alpha.py)
```

---

### 🔹 4. Fine-tuning or Resume

You can resume from any checkpoint by setting `model_s3_key` or `save_path` in your config. Use:

```bash
--resume true
```

---

## 🧪 Evaluation & Testing

Akeso V2 includes a **unified test suite** for Grading, Segmentation, Localization, and Router evaluation.

---

### 🔹 1. Individual Model Testing

Run tests for each task separately to validate checkpoints:

```bash
# Grading
python3 tests/test_grading.py --config config/grading_config.yaml

# Segmentation
python3 tests/test_segmentation.py --config config/segmentation_config.yaml

# Localization
python3 tests/test_localization.py --config config/localization_config.yaml
```

📝 Logs metrics to Weights & Biases (WandB) and console.

---

### 🔹 2. Dynamic Router Testing

Evaluates the unified model with learned routing:

```bash
python3 tests/test_dynamic_model.py --config config/dynamic_config.yaml
```

🧠 Outputs:

* Task-wise metrics
* Routing weights
* Entropy distribution
* Model stability indicators

---

### 🔹 3. Unified Test Launcher

For full evaluation in one go:

```bash
python3 test_suite.py
```

This automatically spawns:

* All 3 expert head evaluations
* Dynamic router evaluation (if checkpoint present)

---

✅ Results are visualized via WandB for:

* Class-wise confusion matrices (Grading)
* Dice & IoU overlays (Segmentation)
* Error histograms (Localization)
* Task selection entropy (Router)

---

## 📊 Final Performance Summary & Results

This section presents the final evaluation of each task and the unified router model based on the test datasets. For a detailed breakdown, see [`model_performance.md`](model_performance.md).

---

### 🔹 1. Grading Model

| Metric             | Value  |
| ------------------ | ------ |
| Accuracy           | 48.16% |
| Weighted F1 Score  | 0.391  |
| Weighted Precision | 0.408  |
| Weighted Recall    | 0.482  |

**Top Class Precision**: No DR → 0.580
⚠️ *Room for improvement in Severe NPDR and PDR classification.*

---

### 🔹 2. Segmentation Model

| Lesion Type         | Dice Score |
| ------------------- | ---------- |
| Microaneurysms (MA) | 0.702      |
| Haemorrhages (HE)   | 0.791      |
| Hard Exudates (EX)  | 0.814      |
| Soft Exudates (SE)  | 0.824      |
| Optic Disc (OD)     | 0.761      |

**Mean Dice Score**: `0.7846`
✅ *Consistently strong performance, especially on EX and SE.*

---

### 🔹 3. Localization Model

| Target | MAE (px) | Error Range (px) |
| ------ | -------- | ---------------- |
| OD     | 5.88     | 1.47 – 19.11     |
| Fovea  | 12.26    | 1.98 – 38.16     |

✅ *OD localization is accurate*
⚠️ *Fovea error range still shows some variance.*

---

### 🟣 4. Dynamic Router Model

| Metric                 | Value                        |
| ---------------------- | ---------------------------- |
| Routing Entropy        | \~1.10                       |
| Performance Loss       | None                         |
| Task Weights @ Epoch 0 | G: 32.4%, S: 31.6%, L: 36.0% |

✅ *Multitask routing remains stable with no degradation*
🧠 *Backed by softmax annealing & entropy regularization.*

---

📂 **For complete logs, visualizations, and cycle-wise training trends**, refer to:

* 📄 [`Model_Performance.md`](model_performance.md)
* 📊 [WandB Dashboard](https://wandb.ai/diganta-dutta-097/akeso-eyecare)

---

## 🚀 Inference Dashboard & Deployment

Akeso V2 includes a **Streamlit-based diagnostic dashboard** that supports real-time inference across:

* 🧑 Patient View
* 🥼 Doctor View
* 🧠 Clinician View

Each view tailors the output using GPT-4o and overlays from task outputs.

---

### 🖼️ Upload & Process Flow

1. **Upload Image(s)**: Single retinal fundus image or a ZIP of images
2. **Select Role**: Patient / Doctor / Clinician
3. **Run Inference**:

   * Disease grading
   * Lesion segmentation
   * OD/Fovea localization
   * GPT-generated clinical report

---

### 🧠 GPT-based Role-aware Explanations

* **Patient**: Simple, friendly explanation of results
* **Doctor**: Summary of condition, risk level, treatment direction
* **Clinician**: Detailed grading, lesion overlays, disc/fovea precision, risks

---

### 🖥️ How to Launch

```bash
streamlit run app.py
```

---

### 💾 Weights & Model Loading

All models are fetched dynamically from **AWS S3** via paths specified in the `.env` and config YAMLs:

```env
AWS_ACCESS_KEY_ID=xxxx
AWS_SECRET_ACCESS_KEY=yyyy
S3_BUCKET=akeso-storage
```

Models loaded:

* `grading.pt`
* `segmentor.pt`
* `localizer_od_fovea_best.pt`
* `dynamic_router.pt`

---

### 🖼️ Visual Outputs

Each inference produces:

* Grad-CAM visual for classification
* Segmentation mask overlay (Red = prediction, Green = ground truth)
* Localization heatmap (OD/Fovea)
* Expanded report using GPT (Azure OpenAI or GPT-4o)

---

🎥 Demo Video

📂 Click here to watch the demo video on Google Drive: 
Part 1: https://drive.google.com/file/d/1lFZdsozCaeU-mD5ZJ1YciLoTtHbqNTUX/view?usp=sharing
Part 2 (Post Routing): https://drive.google.com/file/d/1aZEWBI0ayjN2GwYdGQgfqL4A1lOPTpTg/view?usp=sharing

---

## 🧠 Future Roadmap + Enhancements

Akeso V2 lays the foundation for modular multi-task retinal diagnosis. The following features are planned or underway for **Akeso V3** and beyond:

---

### 🔗 1. RedisAI Integration

* **Purpose**: Cache previous inferences
* **Outcome**: 5–10× faster response time in dashboard

---

### 🧬 2. Clinical Fine-Tuning

* **Method**: LoRA / qLoRA adaptation of backbone models
* **Goal**: Adapt grading/segmentation models to real-world hospital data
* **Tooling**: HuggingFace PEFT + Low-Rank Optimizer pipelines

---

### 🎯 3. Federated Learning Extension

* **Scope**: Train local heads at hospital/clinic level
* **Core**: Router + encoder shared across clients
* **Security**: Model weight merging with privacy filters

---

### 🖼️ 4. Visual Captioning + Retrieval

* **Task**: Add BLIP-2 / Gemini Flash to describe and retrieve retina images
* **Use Cases**: Differential diagnosis, similar patient history

---

### 📱 5. ONNX/TensorRT Export

* **Goal**: Mobile & edge deployment (Raspberry Pi + Jetson Nano)
* **Process**: Quantize and convert ViT + GNN models for speedup

---

### 🧠 6. RAG + Patient History

* **Direction**: Integrate past visits and reports into GPT
* **Infra**: FAISS / Pinecone + prompt chaining with LangChain

---

Here’s the final section:

---

## 📜 License & References

### 🔖 License

This project is released under the **MIT License**
© 2025 — Akeso Eyecare

---

### 📚 References

* Kirillov et al. [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643)
* Dosovitskiy et al. [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929)
* Vaswani et al. [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* Wang et al. [GATv2Conv: How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491)
* OpenAI [GPT-4o](https://openai.com/index/gpt-4o/)
* BLIP-2: [Salesforce BLIP](https://github.com/salesforce/BLIP)
* Microsoft [LoRA and PEFT](https://arxiv.org/abs/2106.09685)
* Google [MedPaLM](https://arxiv.org/abs/2212.13138)

---