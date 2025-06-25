import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from vision_etl.dataset_factory import get_idrid_dataset
from models.grading_model import GradingModel
from agents.embedding_agent import load_encoder
from utils.wandb_logger import WandBLogger
import yaml, os

def test_grading_model():
    with open("config/grading_config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = get_idrid_dataset(["grading"], mode="test")["grading"]
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    encoder = load_encoder(cfg["encoder"], pretrained=False, freeze=False)
    model = GradingModel(encoder).to(device)
    model.load_state_dict(torch.load(cfg["save_path"]))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    acc = accuracy_score(all_labels, all_preds)

    logger = WandBLogger(task_name="grading", cfg=cfg, mode="test")

    logger.log_metrics({
        "test/accuracy": acc,
        "test/precision": report["weighted avg"]["precision"],
        "test/recall": report["weighted avg"]["recall"],
        "test/f1_score": report["weighted avg"]["f1-score"],
        "test/classwise_precision": {str(k): v["precision"] for k, v in report.items() if k.isdigit()},
        "test/classwise_recall": {str(k): v["recall"] for k, v in report.items() if k.isdigit()},
        "test/classwise_f1": {str(k): v["f1-score"] for k, v in report.items() if k.isdigit()}
    })

    logger.finish()
    os.remove(cfg["save_path"])

    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    test_grading_model()