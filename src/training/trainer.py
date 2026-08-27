import os
import time
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import build_loss_fn
from src.training.metrics import compute_classification_metrics, MetricTracker


class Trainer:
    """
    Standard PyTorch Trainer for LOTA (NBC and NGC) architectures.
    Supports Mixed Precision (AMP), Cosine Learning Rate Scheduling,
    Metric Tracking (ACC, AP, AUROC, F1), and Early Stopping.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
        early_stopping_patience: int = 5,
        model_name: str = "lota_model",
        config: Optional[Dict[str, Any]] = None
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.config = config or {}
        
        training_cfg = self.config.get("training", {})
        self.epochs = training_cfg.get("epochs", 30)
        self.patience = early_stopping_patience or training_cfg.get("early_stopping_patience", 5)
        self.use_amp = (use_amp or training_cfg.get("mixed_precision", True)) and (self.device.type == "cuda")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.criterion = criterion or build_loss_fn("bce_with_logits")

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            lr = training_cfg.get("learning_rate", 0.0001)
            wd = training_cfg.get("weight_decay", 1e-5)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.best_val_auroc = 0.0
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [], "train_auroc": [],
            "val_loss": [], "val_acc": [], "val_auroc": []
        }

    def _forward_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        noise_patch = batch["noise_patch"].to(self.device)
        raw_image = batch.get("raw_image", None)
        if raw_image is not None:
            raw_image = raw_image.to(self.device)

        if hasattr(self.model, "extract_image_features") or hasattr(self.model, "attention"):
            # NGC
            return self.model(noise_patch=noise_patch, raw_image=raw_image)
        else:
            # NBC
            return self.model(noise_patch=noise_patch)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        tracker = MetricTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            labels = batch["label"].to(self.device).view(-1, 1)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self._forward_batch(batch).view(-1, 1)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self._forward_batch(batch).view(-1, 1)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            tracker.update(labels, logits, loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return tracker.compute()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Val") -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()

        pbar = tqdm(dataloader, desc=f"[{desc}]", leave=False)
        for batch in pbar:
            labels = batch["label"].to(self.device).view(-1, 1)
            logits = self._forward_batch(batch).view(-1, 1)
            loss = self.criterion(logits, labels)
            tracker.update(labels, logits, loss.item())

        return tracker.compute()

    def fit(self, epochs: Optional[int] = None, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        num_epochs = epochs or self.epochs
        exp_name = experiment_name or self.model_name
        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{exp_name}_best.pth")
        
        print(f"\n[START TRAINING] {exp_name} on {self.device.type.upper()} for {num_epochs} epochs | AMP: {self.use_amp}")
        start_time = time.time()
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["train_auroc"].append(train_metrics["auroc"])

            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader, desc=f"Epoch {epoch} [Val]")
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])
                self.history["val_auroc"].append(val_metrics["auroc"])

                print(
                    f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
                    f"Train Loss: {train_metrics['loss']:.4f}, ACC: {train_metrics['accuracy']*100:.2f}%, AUROC: {train_metrics['auroc']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, ACC: {val_metrics['accuracy']*100:.2f}%, AUROC: {val_metrics['auroc']:.4f}"
                )

                if val_metrics["auroc"] > self.best_val_auroc:
                    self.best_val_auroc = val_metrics["auroc"]
                    epochs_no_improve = 0
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": self.config
                    }, best_checkpoint_path)
                    print(f"  --> Saved new best checkpoint to {best_checkpoint_path}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"[EARLY STOPPING] Triggered after {epoch} epochs (patience={self.patience}).")
                        break
            else:
                print(
                    f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
                    f"Train Loss: {train_metrics['loss']:.4f}, ACC: {train_metrics['accuracy']*100:.2f}%, AUROC: {train_metrics['auroc']:.4f}"
                )

            if self.scheduler is not None:
                self.scheduler.step()

        elapsed = time.time() - start_time
        print(f"[TRAINING COMPLETE] Elapsed: {elapsed:.2f}s | Best Val AUROC: {self.best_val_auroc:.4f}\n")
        return {
            "history": self.history,
            "best_val_auroc": self.best_val_auroc,
            "best_checkpoint_path": best_checkpoint_path,
            "training_time_seconds": elapsed
        }


# Alias
LOTATrainer = Trainer
