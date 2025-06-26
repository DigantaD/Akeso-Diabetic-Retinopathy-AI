Here’s the full, updated `README.md` incorporating:

* ✅ Final benchmark results
* ✅ Comparative table with global models
* ✅ All previous sections retained and refined

---

# 🧐 Akeso: Modular Vision & Language System for Retinal Disease Diagnosis

> **One-liner:**
> A multi-task, modular AI pipeline for retinal fundus image diagnosis that integrates Grading, Segmentation, and Localization with LLM-powered clinical reports and interpretable visual outputs.

---

## ✨ Why Akeso?

Akeso addresses the need for a unified, interpretable, and role-aware diagnostic platform in diabetic retinopathy care:

* 🔹 Jointly performs **disease severity grading**, **lesion segmentation**, and **optic disc/fovea localization**
* 🔹 Combines **SAM's pretrained ViT** with custom decoders and **Graph Neural Networks**
* 🔹 Leverages **GPT-based LLMs** to explain results for **patients, doctors, and clinicians**
* 🔹 Designed to be **extensible**, lightweight, and capable of integrating with future modules (e.g., captioning, retrieval)

---

## 📊 Architecture Diagram

```
                          ┌────────────────────────────────────────────────┐
                          │      👁️  Retinal Fundus Image Input      │
                          └───────────────────────────────────────────────┘
                                           │
                                           ▼
                      ┌────────────────────────────────┐
                      │   Vision Encoder (SAM / ViT) │ ◄──── Pretrained SAM ViT-B
                      └────────────────────────────────┘
                          │             │             │
                          ▼             ▼             ▼
        ┌─────────────────────┐ ┌───────────────┐ ┌──────────────────┐
        │ Disease Grading     │ │ Segmentation    │ │ Localization       │
        │ (MLP Head + GradCAM)│ │ (SAM + Decoder)│ │ (GNN + Heatmaps)  │
        └─────────────────────┘ └───────────────┘ └──────────────────┘
                          │             │             │
                          └────────────────────────────┘
                                 ▼             ▼
                     ┌─────────────────────────────────┐
                     │ GPT-4o (LLM) Clinical Explanation │
                     │ - Role-based reporting           │
                     │ - Prompt Templates               │
                     └─────────────────────────────────┘
```

---

## 🧠 Modeling Approach & Design Choices

### ✅ Shared Vision Backbone (SAM ViT-B)

* Offers pretrained spatial reasoning and strong generalization from large-scale vision training
* Facilitates seamless integration with Segment Anything's modular decoders
* Eliminates the need to pretrain our own ViT from scratch

### ✅ Grading Module

* MLP classification head with GradCAM visualization
* Clinical explainability via class activation heatmaps

### ✅ Segmentation Module

* Combines SAM's mask decoder with a U-Net-style enhancement decoder
* Tailored to capture lesion-specific detail absent in general SAM

### ✅ Localization Module

* Patch-level GNN (GATv2Conv) + decoder to localize optic disc and fovea
* GNN captures spatial dependencies more robustly than regression alone

### ✅ LLM-Powered Report Generator

* GPT-4o with dynamic prompt templates
* Tailors medical explanations based on role: patient, doctor, clinician

---

## 📁 Folder Structure (Simplified)

```
├── agents/              # Embedding agents
├── config/              # Task-wise configs
├── dashboard/           # Streamlit-based multi-role UI
├── models/              # Seg, Grade, Loc models
├── outputs/             # Saved checkpoints (.pt)
├── pretrained/          # Downloaded SAM ViT model
├── segment-anything/    # Git-cloned Segment Anything repo
├── tests/               # Evaluation scripts
├── training/            # Training code + losses
├── utils/               # GradCAM, WandB logger, postprocessing
├── vision_etl/          # Data loaders for all 3 tasks
├── app.py               # Unified Streamlit app entry
├── run_all_tests.py     # Auto-run all test suites
├── trainer.sh           # Script for full training loop
```

---

## ⚙️ Setup Instructions

```bash
# Clone main repo
git clone https://github.com/DigantaD/Akeso-Diabetic-Retinopathy-AI
cd akeso-retina-ai

# Install dependencies
pip install -r requirements.txt

# Segment Anything setup
cd segment-anything && pip install -e . && cd ..

# Download pretrained ViT-B model
mkdir -p pretrained
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P pretrained/
```

---

## 🚀 Training

```bash
# Train individual modules
python training/train_grading.py --config config/grading_config.yaml
python training/train_segmentation.py --config config/segmentation_config.yaml
python training/train_localization.py --config config/localization_config.yaml

# Run all together
bash trainer.sh
```

---

## 🔍 Inference Dashboard

```bash
streamlit run app.py
```

🎥 Demo Video

📂 Click here to watch the demo video on Google Drive: https://drive.google.com/file/d/1lFZdsozCaeU-mD5ZJ1YciLoTtHbqNTUX/view?usp=sharing

**Features:**

* Upload single or zipped retinal images
* View segmentation masks, grading output, localization heatmaps
* GPT-4o clinical report with role selector (Patient/Doctor/Clinician)

---

## 📊 Benchmarks & Evaluation

### 🔬 Akeso Performance (Test Set)

| Task             | Metric                            | Value    |
| ---------------- | --------------------------------- | -------- |
| **Grading**      | Overall Accuracy                  | 49.5%    |
|                  | Weighted F1 Score                 | 47.2%    |
| **Segmentation** | Dice Score (avg across 5 classes) | 0.26     |
|                  | IoU Score (avg across 5 classes)  | 0.22     |
| **Localization** | Avg Pixel Error                   | 85.79 px |

---

### 🌍 Comparison with Global Models

| Model        | Task                | Accuracy / Dice / Error   | Explainability     | Open Source |
| ------------ | ------------------- | ------------------------- | ------------------ | ----------- |
| **Akeso**    | Grading + Seg + Loc | 49.5% / Dice 0.26 / 85 px | ✅ GradCAM + GPT-4o | ✅ Yes       |
| MedPaLM      | Grading             | \~67% (QA only)           | ❌                  | ❌ No        |
| SAM          | Segmentation        | Excellent on objects      | ❌                  | ✅ Yes       |
| Gemini Flash | Captioning/VQA      | ✨ Fast multimodal         | ✅ (VLM outputs)    | ❌ No        |
| BioViL       | Localization        | \~72 px error             | ❌                  | ✅ Yes       |

---

## ⚡ Future Enhancements

* 🔗 RedisAI for faster inference caching
* 🧠 Add BLIP2 or Gemini for multimodal captioning
* 🏛 LoRA-based fine-tuning for clinical personalization
* 🏠 ONNX/TensorRT export for mobile compatibility
* 📦 Vector DB + RAG for patient history integration

---

## 📜 License

MIT License © 2025

---