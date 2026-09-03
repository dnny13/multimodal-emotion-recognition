# -*- coding: utf-8 -*-
"""
evaluation/metrics.py
=====================
Fungsi-fungsi evaluasi metrik klasifikasi untuk unimodal dan multimodal.
Mendukung:
- Metrik standar (akurasi, F1, precision, recall, confusion matrix)
- Masking evaluation (untuk mengukur kontribusi modalitas)
- Geometric Mean Fusion (Lu et al. 2026)
- Test-Time Augmentation (TTA) untuk boost akurasi tanpa training ulang
- Temperature Scaling & KNN Re-ranking (Strategi 3)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score
)
from tqdm import tqdm

from training.utils import load_checkpoint
from models import (
    UnimodalAudioModel,
    UnimodalVisualModel,
    MultimodalFusionModel
)


# ============================================================
# 1. METRIK KLASIFIKASI DASAR
# ============================================================

def compute_classification_metrics(y_true, y_pred):
    """
    Menghitung seluruh metrik klasifikasi.

    Args:
        y_true: list atau array label sebenarnya
        y_pred: list atau array label prediksi

    Returns:
        dict: Seluruh metrik
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    class_counts = cm.sum(axis=1)
    class_acc = cm.diagonal() / (class_counts + 1e-10)

    weighted_acc = (class_acc * class_counts).sum() / (class_counts.sum() + 1e-10)
    unweighted_acc = class_acc.mean()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'weighted_accuracy': weighted_acc,
        'unweighted_accuracy': unweighted_acc,
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': cm,
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'class_accuracy': class_acc.tolist(),
        'class_counts': class_counts.tolist()
    }


# ============================================================
# 2. EVALUASI UNIMODAL
# ============================================================

def evaluate_unimodal_model(model, test_loader, device, model_type='audio'):
    """
    Evaluasi model unimodal pada test set.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_type}"):
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                embeddings = model.extract_features(inputs)

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
                all_embeddings.append(embeddings.cpu().numpy())

    metrics = compute_classification_metrics(all_targets, all_preds)
    embeddings = np.concatenate(all_embeddings, axis=0)

    return metrics, embeddings


# ============================================================
# 3. EVALUASI MULTIMODAL
# ============================================================

def evaluate_multimodal_model(fusion_model, test_loader, device, use_cached=False):
    """
    Evaluasi model multimodal pada test set.
    """
    fusion_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Multimodal"):
            if use_cached:
                audio_emb, visual_emb, targets = batch
                audio_emb = audio_emb.to(device)
                visual_emb = visual_emb.to(device)
                targets = targets.to(device)

                fused = fusion_model.fusion(audio_emb, visual_emb)
                outputs = fusion_model.classifier(fused)
            else:
                audio_input, visual_input, targets = batch
                audio_input = audio_input.to(device)
                visual_input = visual_input.to(device)
                targets = targets.to(device)

                outputs = fusion_model(audio_input, visual_input)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    metrics = compute_classification_metrics(all_targets, all_preds)
    return metrics


# ============================================================
# 4. EVALUASI TEST SET (FUNGSI UTAMA)
# ============================================================

def evaluate_test_set(
    model_path=None,
    model=None,
    model_type='audio',
    test_loader=None,
    device='cuda',
    fusion_type=None,
    use_cached=False,
    num_classes=8
):
    """
    Evaluasi test set untuk unimodal atau multimodal.
    """
    if model is None and model_path is not None:
        if model_type == 'audio':
            model = UnimodalAudioModel(num_classes=num_classes).to(device)
        elif model_type == 'visual':
            model = UnimodalVisualModel(num_classes=num_classes).to(device)
        elif model_type == 'multimodal':
            model = MultimodalFusionModel(
                num_classes=num_classes,
                fusion_type=fusion_type or 'gmu'
            ).to(device)
        else:
            raise ValueError(f"model_type harus 'audio', 'visual', atau 'multimodal'")

        load_checkpoint(model_path, model, device=device)

    elif model is None and model_path is None:
        raise ValueError("Salah satu dari model atau model_path harus diberikan")

    if model_type == 'multimodal':
        metrics = evaluate_multimodal_model(model, test_loader, device, use_cached)
        embeddings = None
    else:
        metrics, embeddings = evaluate_unimodal_model(model, test_loader, device, model_type)

    return {
        'metrics': metrics,
        'embeddings': embeddings,
        'model_type': model_type,
        'model_path': model_path
    }


# ============================================================
# 5. UTILITY: SERIALIZABLE
# ============================================================

def convert_to_serializable(obj):
    """
    Konversi objek numpy ke tipe Python serializable untuk JSON.
    """
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def save_test_metrics(result_path, metrics):
    """Menyimpan metrik test ke file JSON."""
    os.makedirs(result_path, exist_ok=True)
    metrics_serializable = convert_to_serializable(metrics)
    with open(os.path.join(result_path, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"✅ Test metrics saved to {os.path.join(result_path, 'test_metrics.json')}")


# ============================================================
# 6. MASKING EVALUATION
# ============================================================

def evaluate_multimodal_with_masking(fusion_model, test_loader, device):
    """
    Evaluasi unimodal-dari-model-multimodal dengan menol-kan salah satu
    modalitas saat inferensi (TANPA retraining).
    """
    fusion_model.eval()
    results = {}

    for mask_mode in ['full', 'audio_only', 'visual_only']:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                audio_emb, visual_emb, targets = batch
                audio_emb = audio_emb.to(device).clone()
                visual_emb = visual_emb.to(device).clone()

                if mask_mode == 'audio_only':
                    visual_emb = torch.zeros_like(visual_emb)
                elif mask_mode == 'visual_only':
                    audio_emb = torch.zeros_like(audio_emb)

                fused = fusion_model.fusion(audio_emb, visual_emb)
                outputs = fusion_model.classifier(fused)

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy() if torch.is_tensor(targets) else targets)

        metrics = compute_classification_metrics(all_targets, all_preds)
        results[mask_mode] = metrics

        print(f"\n  [Masking: {mask_mode}] Acc: {metrics['accuracy']*100:.2f}% | "
              f"F1-Macro: {metrics['f1_macro']:.4f}")

    return results


def save_masking_evaluation(result_path, masking_results):
    """Simpan hasil masking evaluation ke JSON."""
    os.makedirs(result_path, exist_ok=True)
    serializable = convert_to_serializable(masking_results)
    out_path = os.path.join(result_path, 'masking_evaluation.json')
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ Masking evaluation saved to {out_path}")
    return out_path


# ============================================================
# 7. GEOMETRIC MEAN FUSION (Lu et al. 2026)
# ============================================================

def geometric_mean_fusion(probs_audio: np.ndarray, probs_visual: np.ndarray) -> np.ndarray:
    """
    Geometric mean fusion untuk probabilitas softmax.
    """
    probs_audio = np.clip(probs_audio, 1e-8, 1.0)
    probs_visual = np.clip(probs_visual, 1e-8, 1.0)
    fused = np.sqrt(probs_audio * probs_visual)
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused


def evaluate_geometric_mean(y_true, probs_audio, probs_visual):
    """
    Evaluasi geometric mean fusion.
    """
    fused_probs = geometric_mean_fusion(probs_audio, probs_visual)
    y_pred = np.argmax(fused_probs, axis=1)
    return compute_classification_metrics(y_true, y_pred)


# ============================================================
# 8. TEST-TIME AUGMENTATION (TTA)
# ============================================================

def evaluate_with_tta(
    fusion_model,
    test_loader,
    device,
    num_aug=5,
    use_flip=True,
    use_rotation=True
):
    """
    Evaluasi model dengan Test-Time Augmentation (TTA).
    """
    fusion_model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TTA Evaluation"):
            audio_emb, visual_emb, targets = batch
            audio_emb = audio_emb.to(device)
            visual_emb = visual_emb.to(device)
            targets = targets.to(device)

            fused = fusion_model.fusion(audio_emb, visual_emb)
            outputs = fusion_model.classifier(fused)
            probs = F.softmax(outputs, dim=1)

            for _ in range(num_aug):
                aug_visual = visual_emb.clone()
                if use_flip and np.random.rand() > 0.5:
                    aug_visual = torch.flip(aug_visual, dims=[-1])
                if use_rotation and np.random.rand() > 0.5:
                    k = np.random.randint(1, 4)
                    aug_visual = torch.rot90(aug_visual, k=k, dims=[-2, -1])
                fused_aug = fusion_model.fusion(audio_emb, aug_visual)
                outputs_aug = fusion_model.classifier(fused_aug)
                probs += F.softmax(outputs_aug, dim=1)

            probs /= (num_aug + 1)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    metrics = compute_classification_metrics(all_targets, all_preds)
    print(f"\n📊 TTA Evaluation (n_aug={num_aug}):")
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Macro : {metrics['f1_macro']:.4f}")
    return metrics


# ============================================================
# STRATEGI 3: TEMPERATURE SCALING & KNN RE-RANKING
# ============================================================

def apply_temperature_scaling(logits, temperature=1.5):
    """
    Temperature Scaling: membagi logits dengan T sebelum softmax.
    T > 1: melembutkan prediksi (mengurangi overconfidence).
    T < 1: memperkeras prediksi.
    """
    return F.softmax(logits / temperature, dim=1)


def knn_re_ranking(embeddings, logits, labels, k=5, temperature=1.5):
    """
    KNN Re-ranking: mengoreksi prediksi berdasarkan K tetangga terdekat di embedding space.
    
    Args:
        embeddings: (N, D) array embedding
        logits: (N, C) array logits
        labels: (N,) array label (hanya untuk referensi, tidak digunakan dalam perhitungan)
        k: jumlah tetangga
        temperature: parameter temperature scaling
    
    Returns:
        final_probs: (N, C) array probabilitas setelah re-ranking
    """
    from sklearn.neighbors import NearestNeighbors
    
    # 1. Temperature scaling
    logits_tensor = torch.FloatTensor(logits)
    probs = apply_temperature_scaling(logits_tensor, temperature).numpy()
    
    # 2. Cari K tetangga terdekat di embedding space
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    # 3. Re-rank: rata-rata probabilitas dari tetangga
    reranked_probs = np.zeros_like(probs)
    for i in range(len(probs)):
        neighbor_probs = probs[indices[i]]
        reranked_probs[i] = neighbor_probs.mean(axis=0)
    
    # 4. Kombinasi dengan probabilitas asli (weighted average)
    final_probs = 0.7 * probs + 0.3 * reranked_probs
    return final_probs


def evaluate_with_knn_reranking(
    fusion_model,
    test_loader,
    device,
    k=5,
    temperature=1.5
):
    """
    Evaluasi model multimodal dengan KNN Re-ranking pada test set.
    
    Args:
        fusion_model: MultimodalFusionModel (sudah dilatih)
        test_loader: DataLoader yang mengembalikan (audio_emb, visual_emb, targets)
        device: 'cuda' atau 'cpu'
        k: jumlah tetangga untuk KNN
        temperature: parameter temperature scaling
    
    Returns:
        dict: Metrik klasifikasi setelah KNN re-ranking
    """
    fusion_model.eval()
    all_embeddings = []
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="KNN Re-ranking Evaluation"):
            audio_emb, visual_emb, targets = batch
            audio_emb = audio_emb.to(device)
            visual_emb = visual_emb.to(device)
            targets = targets.to(device)
            
            fused = fusion_model.fusion(audio_emb, visual_emb)
            logits = fusion_model.classifier(fused)
            
            all_embeddings.append(fused.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    
    # KNN Re-ranking
    final_probs = knn_re_ranking(embeddings, logits, all_targets, k=k, temperature=temperature)
    preds = np.argmax(final_probs, axis=1)
    
    metrics = compute_classification_metrics(all_targets, preds)
    print(f"\n📊 KNN Re-ranking Evaluation (k={k}, T={temperature}):")
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Macro : {metrics['f1_macro']:.4f}")
    return metrics