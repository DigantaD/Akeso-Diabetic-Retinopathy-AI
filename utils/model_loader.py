import torch
import os
from dotenv import load_dotenv
from models.router import TaskRouter
from models.main_model import AkesoMultiTaskModel
from models.grading_model import GradingModel
from models.segmenter import AkesoSegmentationModel
from models.localization_model import LocalizationModel
from agents.embedding_agent import load_encoder
from dashboard.inference.s3_model_loader import download_model_from_s3

# Load AWS credentials (if needed elsewhere)
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = "akeso-eyecare"

def load_torch_model(local_path: str):
    """
    Loads a PyTorch checkpoint from a local file path.
    """
    print(f"📥 Loading checkpoint from local path: {local_path}")
    return torch.load(local_path, map_location="cpu")

def load_pretrained_model(cfg):
    """
    Loads the complete Akeso multi-task model:
    - Shared encoder
    - Grading head
    - Segmentation head
    - Localization head
    - Task router (if dynamic model)
    """
    # === 1. Load Shared Encoder ===
    encoder_name = cfg.get("encoder_name") or cfg.get("model_config", {}).get("encoder", "openclip-vit-b-16")
    encoder = load_encoder(encoder_name, pretrained=False, freeze=False)

    # === 2. Load Grading Head ===
    grading_ckpt_path = cfg.get("grading_ckpt", "outputs/checkpoints/grading.pt")
    download_model_from_s3(S3_BUCKET, grading_ckpt_path, grading_ckpt_path)
    grading_model = GradingModel(encoder=None, encoder_dim=encoder.output_dim)
    grading_ckpt = load_torch_model(grading_ckpt_path)
    grading_state = grading_ckpt.get("model_state_dict", grading_ckpt)
    grading_model.load_state_dict(grading_state, strict=False)
    grading_head = grading_model.head

    # === 3. Load Segmentation Head ===
    sam_ckpt_path = cfg.get("sam_ckpt_path") or cfg.get("model_config", {}).get("sam_ckpt_path", "pretrained/sam_vit_b_01ec64.pth")
    sam_model_type = cfg.get("sam_model_type") or cfg.get("model_config", {}).get("sam_model_type", "vit_b")
    segmentation_ckpt_path = cfg.get("segmentation_ckpt", "outputs/checkpoints/segmentor.pt")
    download_model_from_s3(S3_BUCKET, segmentation_ckpt_path, segmentation_ckpt_path)
    segmentation_model = AkesoSegmentationModel(sam_ckpt_path=sam_ckpt_path, model_type=sam_model_type)
    segmentation_ckpt = load_torch_model(segmentation_ckpt_path)
    segmentation_state = segmentation_ckpt.get("model_state_dict", segmentation_ckpt)
    segmentation_model.load_state_dict(segmentation_state, strict=False)
    segmentation_head = segmentation_model

    # === 4. Load Localization Head ===
    localization_ckpt_path = cfg.get("localization_ckpt", "outputs/checkpoints/localizer_od_fovea_best.pt")
    use_vlm = cfg.get("model_config", {}).get("vlm_model", None) is not None
    download_model_from_s3(S3_BUCKET, localization_ckpt_path, localization_ckpt_path)
    localization_model = LocalizationModel(use_vlm_head=True)
    localization_ckpt = load_torch_model(localization_ckpt_path)
    localization_state = localization_ckpt.get("model_state_dict", localization_ckpt)
    localization_model.load_state_dict(localization_state, strict=False)
    localization_head = localization_model

    # === 5. Load Task Router (if dynamic config) ===
    model = AkesoMultiTaskModel(
        encoder=encoder,
        task_heads=[grading_head, segmentation_head, localization_head]
    )

    if "save_path" in cfg and "router" not in cfg:
        # dynamic_router checkpoint might contain router separately
        router_ckpt = load_torch_model(cfg["save_path"])
        download_model_from_s3(S3_BUCKET, cfg["save_path"], cfg["save_path"])
        if isinstance(router_ckpt, dict) and "router_state_dict" in router_ckpt:
            router = TaskRouter(in_dim=encoder.output_dim, num_tasks=3)
            router.load_state_dict(router_ckpt["router_state_dict"])
            model.router = router
    else:
        # fallback: random router for inference
        model.router = TaskRouter(in_dim=encoder.output_dim, num_tasks=3)

    print("✅ Akeso Multi-task model successfully loaded.")
    return model