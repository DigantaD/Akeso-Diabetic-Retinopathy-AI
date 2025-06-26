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

**Why SAM ViT-B?**

* Offers pretrained spatial reasoning and strong generalization from large-scale vision training.
* Facilitates seamless integration with Segment Anything's modular decoders.
* Eliminates the need to pretrain our own ViT from scratch for segmentation/localization.

### ✅ Grading Module

* MLP classification head with GradCAM visualization for interpretable disease severity classification.
* Why? Clinical explainability is crucial. GradCAM overlays help visualize retinal zones influencing the decision.

### ✅ Segmentation Module

* Combines SAM's mask decoder with a U-Net-style enhancement decoder.
* Why? SAM provides base instance awareness, but custom heads improve lesion-level fine-grained segmentation.

### ✅ Localization Module

* Uses patch-level GNN (GATv2Conv) + decoder to localize optic disc and fovea.
* Why? Coordinates alone are insufficient in fundus tasks. GNNs capture spatial patch dependencies for precise localization.

### ✅ LLM-Powered Report Generator

* GPT-4o based prompt templates produce role-specific explanations (patient, doctor, clinician).
* Why? Reports need to be adaptive, empathetic, and medically accurate for different user types.

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
$ git clone https://github.com/your-org/akeso-retina-ai.git
$ cd akeso-retina-ai

# Install core dependencies
$ pip install -r requirements.txt

# Install segment-anything
$ cd segment-anything && pip install -e . && cd ..

# Download pretrained SAM ViT model
$ mkdir -p pretrained
$ wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P pretrained/
```

---

## 🚀 Training

```bash
# Train individual modules
python training/train_grading.py --config config/grading_config.yaml
python training/train_segmentation.py --config config/segmentation_config.yaml
python training/train_localization.py --config config/localization_config.yaml

# Or run all via bash
bash trainer.sh
```

---

## 🔍 Inference Dashboard

```bash
streamlit run app.py
```

**Dashboard Features:**

* Upload image(s) or .zip
* View:

  * Disease severity
  * Lesion masks
  * Optic disc/fovea heatmaps
  * GPT-4o report (role-aware)
* Select viewer mode: Patient / Doctor / Clinician

---

## 📊 Benchmarks (coming soon)

*You will be able to compare Akeso with global public models (MedPaLM, SAM, Gemini Flash, etc.) on metrics like accuracy, Dice score, localization error, explainability.*

---

## ⚡ Future Enhancements

* 🔗 Integrate RedisAI for faster inference caching
* 🧠 Add BLIP2/Gemini for caption-based diagnosis
* 🏛 LoRA fine-tuning for lightweight deployment
* 🏠 ONNX export for edge/mobile compatibility
* 🌟 Add support for longitudinal tracking + RAG memory

---

## 📜 License

MIT License. (c) 2025