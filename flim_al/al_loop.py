"""
ALLoop — Active Learning loop for FLIM decoders
================================================

Implements the iterative AL cycle:
    1. Start with small labeled pool L (seed images)
    2. Train UncertaintyDecoder on L
    3. Score unlabeled pool U with acquisition function
    4. Query oracle for top-K labels → move to L
    5. Repeat until budget exhausted or stopping criterion met

Oracle mode: 'simulated' (uses ground-truth masks on disk)
             'interactive' (prompts user to annotate)

Usage example:
    from flim_al import ALLoop, UncertaintyDecoder
    from flim_al.dataset import FLIMDataset  # or any PyTorch Dataset

    al = ALLoop(
        model=UncertaintyDecoder(base_decoder),
        labeled_dataset=...,
        unlabeled_dataset=...,
        acquisition='entropy',
        query_size=5,
        n_rounds=10,
        device='cuda',
    )
    results = al.run()
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .acquisition import rank_pool
from .uncertainty_decoder import UncertaintyDecoder


class ALLoop:
    """
    Parameters
    ----------
    model           : UncertaintyDecoder (or plain nn.Module)
    labeled_dataset : Dataset with __getitem__ → (image, mask, idx)
    unlabeled_dataset: Same dataset without masks (or same with mask=None)
    acquisition     : 'entropy' | 'least_confidence' | 'bald'
    query_size      : K — images to label per round
    n_rounds        : total AL rounds
    n_epochs        : training epochs per round
    lr              : optimizer learning rate
    device          : 'cuda' | 'cpu'
    save_dir        : directory to save checkpoints and metrics
    train_fn        : optional custom training function(model, loader, epochs, device)
    """

    def __init__(
        self,
        model: nn.Module,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        acquisition: str = "entropy",
        query_size: int = 5,
        n_rounds: int = 10,
        n_epochs: int = 20,
        lr: float = 1e-3,
        device: str = "cuda",
        save_dir: str = "out/al_results",
        train_fn: Callable | None = None,
    ):
        self.model = model.to(device)
        self.labeled_ds = labeled_dataset
        self.unlabeled_ds = unlabeled_dataset
        self.acquisition = acquisition
        self.query_size = query_size
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.save_dir = Path(save_dir)
        self.train_fn = train_fn or self._default_train

        # Index pools
        self.labeled_indices: list[int] = list(range(len(labeled_dataset)))
        self.unlabeled_indices: list[int] = list(range(len(unlabeled_dataset)))

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    # ─── Training ────────────────────────────────────────────────────────────

    def _default_train(
        self,
        model: nn.Module,
        loader: DataLoader,
        n_epochs: int,
        device: str,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        last_loss = 0.0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in loader:
                imgs, masks = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                out = model(imgs)
                if isinstance(out, tuple):
                    seg, log_var = out
                    loss = UncertaintyDecoder.nll_loss(seg, log_var, masks.float())
                else:
                    loss = nn.functional.binary_cross_entropy(
                        out.clamp(1e-6, 1 - 1e-6), masks.float()
                    )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            last_loss = epoch_loss / max(len(loader), 1)
        return last_loss

    # ─── Acquisition ─────────────────────────────────────────────────────────

    def _score_pool(self) -> tuple[list[int], list[float]]:
        pool_tensors = []
        for idx in self.unlabeled_indices:
            sample = self.unlabeled_ds[idx]
            img = sample[0] if isinstance(sample, (list, tuple)) else sample
            pool_tensors.append(img.unsqueeze(0))

        ranked, scores = rank_pool(
            self.model,
            pool_tensors,
            device=self.device,
            method=self.acquisition,
        )
        # Map local pool rank → original unlabeled index
        ranked_global = [self.unlabeled_indices[r] for r in ranked]
        return ranked_global, scores

    # ─── Main loop ───────────────────────────────────────────────────────────

    def run(self) -> list[dict]:
        print(f"\n{'='*60}")
        print(f"  AL Loop | {self.n_rounds} rounds | query_size={self.query_size}")
        print(f"  acquisition={self.acquisition} | device={self.device}")
        print(f"{'='*60}")

        for round_idx in range(self.n_rounds):
            t0 = time.time()
            print(f"\n[Round {round_idx+1}/{self.n_rounds}] "
                  f"labeled={len(self.labeled_indices)} "
                  f"unlabeled={len(self.unlabeled_indices)}")

            # ── Train on current labeled pool ──────────────────────────────
            labeled_subset = Subset(self.labeled_ds, self.labeled_indices)
            loader = DataLoader(labeled_subset, batch_size=4, shuffle=True)
            loss = self.train_fn(self.model, loader, self.n_epochs, self.device)
            print(f"  train_loss={loss:.4f}")

            # ── Evaluate (Dice/F1 on labeled pool as proxy) ─────────────
            dice = self._eval_dice(loader)
            print(f"  proxy_dice={dice:.4f}")

            # ── Acquisition: score unlabeled pool ─────────────────────────
            if not self.unlabeled_indices:
                print("  Unlabeled pool exhausted — stopping early.")
                break

            ranked_global, scores = self._score_pool()
            selected = ranked_global[: self.query_size]
            top_score = scores[: self.query_size]
            print(f"  selected_ids={selected}  top_scores={[f'{s:.3f}' for s in top_score]}")

            # ── Oracle query (simulated: move to labeled pool) ─────────────
            for idx in selected:
                self.labeled_indices.append(idx)
                self.unlabeled_indices.remove(idx)

            # ── Log ───────────────────────────────────────────────────────
            entry = {
                "round": round_idx + 1,
                "labeled_size": len(self.labeled_indices),
                "train_loss": round(loss, 5),
                "proxy_dice": round(dice, 4),
                "selected_ids": selected,
                "elapsed_s": round(time.time() - t0, 1),
            }
            self.history.append(entry)
            self._save_checkpoint(round_idx)

        self._save_history()
        print(f"\n[ALLoop] Done. Results saved to {self.save_dir}")
        return self.history

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_dice(self, loader: DataLoader) -> float:
        self.model.eval()
        dices = []
        for batch in loader:
            imgs, masks = batch[0].to(self.device), batch[1].to(self.device)
            out = self.model(imgs)
            pred = out[0] if isinstance(out, tuple) else out
            pred_bin = (pred > 0.5).float()
            intersection = (pred_bin * masks).sum(dim=[1, 2, 3])
            union = pred_bin.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3])
            dice = (2 * intersection / (union + 1e-8)).mean().item()
            dices.append(dice)
        self.model.train()
        return float(torch.tensor(dices).mean())

    def _save_checkpoint(self, round_idx: int):
        path = self.save_dir / f"model_round{round_idx+1:02d}.pt"
        torch.save(self.model.state_dict(), path)

    def _save_history(self):
        path = self.save_dir / "al_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  History → {path}")
