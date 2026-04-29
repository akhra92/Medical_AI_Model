"""
Training loop with:
  - Weighted cross-entropy loss (class imbalance)
  - AdamW optimizer + CosineAnnealingLR scheduler
  - Early stopping
  - Two-stage training for multimodal/image models:
      Stage 1: Freeze backbone, train head (5 epochs warm-up)
      Stage 2: Unfreeze backbone, train end-to-end with lower LR
  - Per-epoch metric logging
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    CLASS_WEIGHTS, NUM_CLASSES, CHECKPOINT_DIR,
    PATIENCE,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_class_weights(device: torch.device) -> torch.Tensor:
    return torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """
    Handles the full training loop for any of the three model modes.

    Args:
        model: nn.Module (ImageModel | TabularModel | MultimodalModel)
        mode: "image" | "tabular" | "multimodal"
        fold: fold index (used for checkpoint naming)
    """

    def __init__(self, model: nn.Module, mode: str = "multimodal", fold: int = 0):
        self.device = get_device()
        self.model = model.to(self.device)
        self.mode = mode
        self.fold = fold

        # Weighted loss for class imbalance
        self.criterion = nn.CrossEntropyLoss(
            weight=get_class_weights(self.device)
        )

        # Optimizer (set up properly for two-stage training)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0
        self.history: dict = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        """Run one epoch. Returns (avg_loss, accuracy)."""
        self.model.train(training)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
                loss, preds, labels = self._forward_batch(batch)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def _forward_batch(self, batch):
        """Handle all three modes uniformly. Returns (loss, preds, labels)."""
        if self.mode == "multimodal":
            images, tabular, labels = batch
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images, tabular)
        elif self.mode == "image":
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
        elif self.mode == "tabular":
            tabular, labels = batch
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(tabular)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        warmup_epochs: int = 5,
    ) -> dict:
        """
        Full training loop with optional backbone warmup.

        warmup_epochs: epochs to train with backbone frozen (only for image/multimodal)
        Returns history dict.
        """
        print(f"\n{'='*60}")
        print(f"  Fold {self.fold} | Mode: {self.mode} | Device: {self.device}")
        print(f"{'='*60}")

        # ── Stage 1: Warm-up (freeze backbone) ────────────────────────────────
        if self.mode in ("image", "multimodal") and warmup_epochs > 0:
            print(f"\n[Stage 1] Warming up for {warmup_epochs} epochs (backbone frozen)")
            self._freeze_backbone()
            self._reset_optimizer(lr=LEARNING_RATE * 5)   # higher LR for head only
            for epoch in range(warmup_epochs):
                self._step(epoch, train_loader, val_loader, stage="warmup")

        # ── Stage 2: End-to-end fine-tuning ───────────────────────────────────
        print(f"\n[Stage 2] End-to-end training for {NUM_EPOCHS} epochs")
        self._unfreeze_backbone()
        self._reset_optimizer(lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )
        self.patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            stop = self._step(epoch, train_loader, val_loader, stage="finetune")
            if stop:
                break

        # Restore best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nRestored best model (val_loss={self.best_val_loss:.4f})")

        return self.history

    def _step(self, epoch: int, train_loader, val_loader, stage: str) -> bool:
        """One epoch of train + val. Returns True if early-stopping triggered."""
        t0 = time.time()
        train_loss, train_acc = self._run_epoch(train_loader, training=True)
        val_loss, val_acc = self._run_epoch(val_loader, training=False)
        self.scheduler.step()

        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"  [{stage}] Epoch {epoch+1:3d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={lr:.2e} | {time.time()-t0:.1f}s"
        )

        # Early stopping & checkpointing
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.patience_counter = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_fold{self.fold}_{self.mode}.pt")
            torch.save(self.best_model_state, ckpt_path)
        else:
            self.patience_counter += 1
            if self.patience_counter >= PATIENCE:
                print(f"  Early stopping triggered (patience={PATIENCE})")
                return True

        return False

    def _freeze_backbone(self):
        if self.mode == "image":
            self.model.freeze_backbone()
        elif self.mode == "multimodal":
            self.model.freeze_image_backbone()

    def _unfreeze_backbone(self):
        if self.mode == "image":
            self.model.unfreeze_backbone()
        elif self.mode == "multimodal":
            self.model.unfreeze_image_backbone()

    def _reset_optimizer(self, lr: float):
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY,
        )

    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on a DataLoader.
        Returns (y_true, y_pred, y_proba) all as numpy arrays.
        """
        self.model.eval()
        all_labels, all_preds, all_probas = [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference"):
                logits, labels = self._get_logits(batch, return_labels=True)
                preds = logits.argmax(dim=1)
                probas = torch.softmax(logits, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probas.extend(probas)

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probas),
        )

    def _forward_batch_eval(self, batch):
        """Like _forward_batch but no loss computation."""
        if self.mode == "multimodal":
            images, tabular, labels = batch
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images, tabular)
        elif self.mode == "image":
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
        else:
            tabular, labels = batch
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(tabular)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def _get_logits(self, batch, return_labels: bool = False):
        if self.mode == "multimodal":
            images, tabular, labels = batch
            logits = self.model(images.to(self.device), tabular.to(self.device))
        elif self.mode == "image":
            images, labels = batch
            logits = self.model(images.to(self.device))
        else:
            tabular, labels = batch
            logits = self.model(tabular.to(self.device))
        if return_labels:
            return logits, labels.to(self.device)
        return logits
