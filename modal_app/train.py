"""
Training loops for K-VRC animation heads.

Head 1 (ClipRetrievalHead): trained on Stream C pseudo-labels via cosine similarity
Head 2 (MDMRefinementHead): trained on Stream C keypoint sequences

Usage:
    python3 -c "from modal_app.train import train_head1; train_head1()"
    python3 -c "from modal_app.train import train_head2; train_head2()"
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from modal_app.encoder import embed
from modal_app.heads import ClipRetrievalHead, MDMRefinementHead, apply_delta_quaternions
from modal_app.pipeline.bake_clips import compute_clip_similarity, load_baked_clip_keypoints

DATA_PATH = "data/processed"
MODEL_PATH = "data/models"


# ── Head 1: Clip Retrieval ────────────────────────────────────────────────────

def train_head1(
    n_epochs: int = 50,
    lr: float = 1e-3,
    confidence_threshold: float = 0.7,
):
    """Train clip retrieval head on Stream C pseudo-labels."""
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    # Load processed data
    keypoints = np.load(f"{DATA_PATH}/keypoints.npy")  # [N, 60, 6, 2]
    with open(f"{DATA_PATH}/texts.json") as f:
        texts = json.load(f)

    print(f"Loaded {len(texts)} records")

    # Load baked clip reference keypoints
    ref_kps, clip_names = load_baked_clip_keypoints(f"{DATA_PATH}/clip_keypoints.json")
    n_clips = len(clip_names)
    print(f"Loaded {n_clips} reference clips")

    # Save clip name order for inference
    with open(f"{MODEL_PATH}/clip_names.json", "w") as f:
        json.dump(clip_names, f)

    # Generate pseudo-labels: find the best-matching clip for each observed sequence
    print("Generating pseudo-labels via cosine similarity...")
    pseudo_labels, valid_indices = [], []
    for i, kp in enumerate(keypoints):
        sims = compute_clip_similarity(kp, ref_kps)
        best_clip = int(sims.argmax())
        if sims[best_clip] >= confidence_threshold:
            pseudo_labels.append(best_clip)
            valid_indices.append(i)

    print(f"Valid records after confidence filter: {len(valid_indices)}/{len(keypoints)}")
    if len(valid_indices) == 0:
        print("WARNING: No valid records found. Lowering confidence_threshold may help.")
        print("Falling back to all records with best-match labels...")
        for i, kp in enumerate(keypoints):
            sims = compute_clip_similarity(kp, ref_kps)
            pseudo_labels.append(int(sims.argmax()))
            valid_indices.append(i)

    # Embed texts
    valid_texts = [texts[i] for i in valid_indices]
    print(f"Embedding {len(valid_texts)} texts...")
    text_embeddings = embed(valid_texts)  # [N_valid, 384]

    # Train
    head = ClipRetrievalHead(n_clips=n_clips)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    X = torch.from_numpy(text_embeddings)
    Y = torch.tensor(pseudo_labels, dtype=torch.long)

    print(f"Training Head 1 for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        head.train()
        optimizer.zero_grad()
        logits = head.proj(X)  # [N, n_clips] raw logits (pre-softmax)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            acc = (logits.argmax(dim=1) == Y).float().mean()
            print(f"Epoch {epoch+1}/{n_epochs} — loss: {loss.item():.4f}, acc: {acc:.3f}")

    torch.save(head.state_dict(), f"{MODEL_PATH}/clip_head.pt")
    print(f"Saved Head 1 → {MODEL_PATH}/clip_head.pt")


# ── Head 2: MDM Refinement ────────────────────────────────────────────────────

def keypoints_to_pseudo_quaternions(kp_seq: np.ndarray) -> np.ndarray:
    """
    Convert 2D keypoint sequence [60, 6, 2] to pseudo target quaternions [60, 6, 4].
    Encodes relative joint positions as small rotations around identity.
    """
    T, J, _ = kp_seq.shape
    quats = np.zeros((T, J, 4), dtype=np.float32)
    quats[:, :, 3] = 1.0  # start from identity quaternion (w=1)

    # Head joint: encode x-position deviation as small x-axis tilt
    head_x = kp_seq[:, 0, 0]  # normalized x in [-1, 1]
    quats[:, 0, 0] = head_x * 0.2
    quats[:, 0, 3] = np.sqrt(np.maximum(0.0, 1.0 - (head_x * 0.2) ** 2))

    # Normalize all quaternions
    norms = np.linalg.norm(quats, axis=-1, keepdims=True).clip(min=1e-8)
    return quats / norms


def train_head2(
    n_epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 32,
    delta_regularization: float = 0.01,
):
    """Train MDM refinement head on Stream C keypoint sequences."""
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    keypoints = np.load(f"{DATA_PATH}/keypoints.npy")  # [N, 60, 6, 2]
    with open(f"{DATA_PATH}/texts.json") as f:
        texts = json.load(f)

    print(f"Loaded {len(texts)} records")

    print("Generating target quaternions from keypoints...")
    target_quats = np.stack([keypoints_to_pseudo_quaternions(kp) for kp in keypoints])
    # [N, 60, 6, 4]

    print(f"Embedding {len(texts)} texts...")
    embeddings = embed(texts)  # [N, 384]

    head = MDMRefinementHead(n_joints=6, n_frames=60)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    X = torch.from_numpy(embeddings)
    Y = torch.from_numpy(target_quats)
    base = torch.zeros_like(Y)
    base[:, :, :, 3] = 1.0  # identity base rotations

    N = len(X)
    print(f"Training Head 2 for {n_epochs} epochs (N={N}, batch_size={batch_size})...")

    for epoch in range(n_epochs):
        head.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            ctx, tgt, b = X[idx], Y[idx], base[idx]
            optimizer.zero_grad()
            deltas = head(ctx, b)
            pred = apply_delta_quaternions(b, deltas)

            recon_loss = F.mse_loss(pred, tgt)
            reg_loss = deltas.norm(dim=-1).mean()
            loss = recon_loss + delta_regularization * reg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} — loss: {total_loss / n_batches:.5f}")

    torch.save(head.state_dict(), f"{MODEL_PATH}/mdm_head.pt")
    print(f"Saved Head 2 → {MODEL_PATH}/mdm_head.pt")
