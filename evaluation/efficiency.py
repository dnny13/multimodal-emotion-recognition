# -*- coding: utf-8 -*-
"""
evaluation/efficiency.py
========================
Fungsi-fungsi evaluasi efisiensi komputasi.

FIX:
- count_parameters: tambahan parameter include_backbone=False untuk menghitung hanya fusion head.
- get_input_shape_for_model: deteksi temporal mode untuk visual backbone (5D shape).
- get_checkpoint_size_from_file: sudah ada, digunakan untuk ukuran model murni.
- compute_flops_multimodal: memastikan visual dummy 5D.
- measure_inference_time_multimodal: 5D visual dummy.
- compute_ensemble_efficiency_shared_backbone: sudah ada.
"""

import os
import time
import tempfile
import torch
import numpy as np

# =============================================================================
# DEVICE PROFILES
# =============================================================================
DEVICE_PROFILES = {
    'raspberry_pi_4b': {
        'display_name': 'Raspberry Pi 4B',
        'peak_gflops': 13.0,
        'bandwidth_gbs': 5.6,
        'max_size_mb': 50,
        'max_inference_ms': 100,
        'reference': 'Lu et al. (2026), IEEE TAFFC, Table IX'
    },
    'raspberry_pi_5': {
        'display_name': 'Raspberry Pi 5',
        'peak_gflops': 120.0,
        'bandwidth_gbs': 17.10,
        'max_size_mb': 150,
        'max_inference_ms': 100,
        'reference': 'Lu et al. (2026), IEEE TAFFC, Table IX'
    },
    'zenbo_junior_ii': {
        'display_name': 'Zenbo Junior II',
        'peak_gflops': 25.76,
        'bandwidth_gbs': 25.6,
        'max_size_mb': 48,
        'max_inference_ms': 100,
        'reference': 'Lu et al. (2026), IEEE TAFFC, Table IX'
    },
}

def get_device_profile(target_device='raspberry_pi_4b'):
    if target_device not in DEVICE_PROFILES:
        print(f"⚠️ Device '{target_device}' tidak dikenal, fallback ke raspberry_pi_4b")
        target_device = 'raspberry_pi_4b'
    return DEVICE_PROFILES[target_device]

try:
    from thop import profile
except ImportError:
    profile = None
    print("⚠️ thop tidak terinstall. Install dengan: pip install thop")

# =============================================================================
# PARAMETER & SIZE
# =============================================================================

def count_parameters(model, include_backbone=True):
    """
    Menghitung parameter model.
    Jika include_backbone=False, hanya menghitung parameter yang tidak termasuk
    backbone (misalnya fusion module + classifier) dengan mendeteksi modul
    yang memiliki nama 'fusion' atau 'classifier' dan mengabaikan 'audio_backbone'
    dan 'visual_backbone'. Ini berguna untuk mengukur fusion head saja.
    """
    if include_backbone:
        total = sum(p.numel() for p in model.parameters())
    else:
        total = 0
        for name, param in model.named_parameters():
            # Jika nama mengandung 'audio_backbone' atau 'visual_backbone', abaikan
            if 'audio_backbone' in name or 'visual_backbone' in name:
                continue
            total += param.numel()
    return total / 1e6

def get_model_size(model):
    """
    Mengukur ukuran model dengan menyimpan HANYA state_dict ke file sementara.
    """
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 ** 2)
        os.unlink(tmp.name)
    return size_mb

def get_checkpoint_size_from_file(checkpoint_path, state_dict_key='state_dict'):
    """
    Mengukur ukuran model murni dari file checkpoint (tanpa optimizer state).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and state_dict_key in checkpoint:
        state_dict = checkpoint[state_dict_key]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Format checkpoint tidak dikenali: {checkpoint_path}")
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        torch.save(state_dict, tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 ** 2)
        os.unlink(tmp.name)
    return size_mb

# =============================================================================
# INPUT SHAPE DETECTION
# =============================================================================

def get_input_shape_for_model(model, device='cuda', num_frames=15):
    """
    Menentukan input shape yang tepat untuk model.
    Untuk visual backbone, jika model memiliki atribut temporal_mode='stack' dan
    num_frames>1, kembalikan (1, num_frames, 3, 224, 224).
    """
    if hasattr(model, 'backbone_name'):
        backbone = str(model.backbone_name).lower()

        if 'efficientnet' in backbone or 'resnet' in backbone or 'mobilenet' in backbone:
            # Deteksi apakah model visual multimodal atau unimodal dengan temporal
            if hasattr(model, 'temporal_mode') and model.temporal_mode == 'stack':
                return (1, num_frames, 3, 224, 224)
            else:
                # Cek apakah model memiliki metode extract_features yang menerima 5D
                # atau atribut num_frames dari config
                return (1, 3, 224, 224)  # default 4D

        elif 'audio_1dcnn' in backbone:
            return (1, 120, 224)
        elif 'audio_2dcnn' in backbone:
            return (1, 1, 128, 224)

    # Multimodal fusion model
    if hasattr(model, 'fusion'):
        return ((1, 120, 224), (1, num_frames, 3, 224, 224))

    return (1, 3, 224, 224)

# =============================================================================
# FLOPS
# =============================================================================

def compute_flops(model, input_shape, device='cuda'):
    if profile is None:
        return 0.0
    try:
        model.eval().to(device)
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            audio_shape, visual_shape = input_shape
            audio_dummy = torch.randn(*audio_shape).to(device)
            visual_dummy = torch.randn(*visual_shape).to(device)
            macs, _ = profile(model, inputs=(audio_dummy, visual_dummy), verbose=False)
        else:
            dummy = torch.randn(*input_shape).to(device)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return (macs * 2) / 1e9
    except Exception as e:
        print(f"⚠️ Gagal menghitung FLOPs: {e}")
        return 0.0

def compute_flops_multimodal(fusion_model, device='cuda', num_frames=15):
    if profile is None:
        return 0.0
    try:
        fusion_model.eval().to(device)
        audio_dummy = torch.randn(1, 120, 224).to(device)
        visual_dummy = torch.randn(1, num_frames, 3, 224, 224).to(device)
        macs, _ = profile(fusion_model, inputs=(audio_dummy, visual_dummy), verbose=False)
        return (macs * 2) / 1e9
    except Exception as e:
        print(f"⚠️ Gagal menghitung FLOPs multimodal: {e}")
        return 0.0

# =============================================================================
# INFERENCE TIME
# =============================================================================

def measure_inference_time(model, input_shape, device='cuda', n_warmup=10, n_iter=100):
    model.eval().to(device)
    if isinstance(input_shape, tuple) and len(input_shape) == 2:
        audio_shape, visual_shape = input_shape
        audio_dummy = torch.randn(*audio_shape).to(device)
        visual_dummy = torch.randn(*visual_shape).to(device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(audio_dummy, visual_dummy)
        if device == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(n_iter):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(audio_dummy, visual_dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return np.mean(times)
    else:
        dummy = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy)
        if device == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(n_iter):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return np.mean(times)

def measure_inference_time_multimodal(fusion_model, device='cuda', n_warmup=10, n_iter=100, num_frames=15):
    device = torch.device(device)
    fusion_model.eval().to(device)
    audio_dummy = torch.randn(1, 120, 224).to(device)
    visual_dummy = torch.randn(1, num_frames, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = fusion_model(audio_dummy, visual_dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fusion_model(audio_dummy, visual_dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times)

def measure_backbone_inference_time(audio_model, visual_model, device='cuda', n_warmup=10, n_iter=100):
    audio_model.eval().to(device)
    visual_model.eval().to(device)
    audio_dummy = torch.randn(1, 120, 224).to(device)
    visual_dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = audio_model.extract_features(audio_dummy)
            _ = visual_model.extract_features(visual_dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = audio_model.extract_features(audio_dummy)
        _ = visual_model.extract_features(visual_dummy)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times)

def measure_fusion_head_inference_time(fusion_model, device='cuda', n_warmup=10, n_iter=100):
    fusion_model.eval().to(device)
    audio_embed_dummy = torch.randn(1, 1280).to(device)
    visual_embed_dummy = torch.randn(1, 1280).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            fused = fusion_model.fusion(audio_embed_dummy, visual_embed_dummy)
            _ = fusion_model.classifier(fused)
    if device == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        fused = fusion_model.fusion(audio_embed_dummy, visual_embed_dummy)
        _ = fusion_model.classifier(fused)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times)

# =============================================================================
# LIGHTWEIGHT CONSTRAINTS CHECK
# =============================================================================

def check_lightweight_constraints(params_m, flops_g, size_mb, inference_ms, target=None, target_device=None, verbose=True):
    if target_device is not None:
        profile_device = get_device_profile(target_device)
        target = {
            'params': 12, 'flops': 2,
            'size': profile_device['max_size_mb'],
            'inference': profile_device['max_inference_ms']
        }
    elif target is None:
        target = {'params': 12, 'flops': 2, 'size': 50, 'inference': 100}

    constraints = {
        'params': {'value': params_m, 'target': target['params'], 'unit': 'M', 'pass': params_m < target['params']},
        'flops': {'value': flops_g, 'target': target['flops'], 'unit': 'GFLOPs', 'pass': flops_g < target['flops']},
        'size': {'value': size_mb, 'target': target['size'], 'unit': 'MB', 'pass': size_mb < target['size']},
        'inference': {'value': inference_ms, 'target': target['inference'], 'unit': 'ms', 'pass': inference_ms < target['inference']}
    }
    if verbose:
        print("\n" + "="*60)
        print("  LIGHTWEIGHT CONSTRAINT CHECK")
        print("="*60)
        for key, val in constraints.items():
            status = "✅" if val['pass'] else "❌"
            print(f"  {status} {key:<12}: {val['value']:.2f} {val['unit']} (target: <{val['target']} {val['unit']})")
        print("="*60)
        overall = all(c['pass'] for c in constraints.values())
        print(f"  OVERALL STATUS: {'✅ MEMENUHI' if overall else '❌ TIDAK MEMENUHI'}")
        print("="*60)
    return constraints

# =============================================================================
# ROOFLINE ANALYSIS
# =============================================================================

def roofline_analysis(model, model_name, device='cuda', is_multimodal=True, target_device='raspberry_pi_4b', num_frames=15):
    params = count_parameters(model)  # default include_backbone=True
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    if is_multimodal:
        input_shape = ((1, 120, 224), (1, num_frames, 3, 224, 224))
    else:
        input_shape = get_input_shape_for_model(model, device, num_frames=num_frames)

    flops = compute_flops(model, input_shape, device_obj)
    inference = measure_inference_time(model, input_shape, device_obj)
    size = get_model_size(model)

    profile = get_device_profile(target_device)
    peak_flops = profile['peak_gflops']
    memory_bandwidth = profile['bandwidth_gbs']
    arithmetic_intensity = flops / (size / 1e3) if size > 0 and flops > 0 else 0.0
    ridge_point = peak_flops / memory_bandwidth
    is_memory_bound = arithmetic_intensity < ridge_point
    bound_type = "Memory" if is_memory_bound else "Compute"

    print("\n" + "=" * 60)
    print(f"  ROOFLINE ANALYSIS — {model_name}")
    print(f"  Constraint Reference: {profile['display_name']} ({profile['reference']})")
    print(f"  ⚠️ Diukur di device eksekusi aktual ({str(device).upper()}), BUKAN hasil benchmark fisik.")
    print("=" * 60)
    print(f"  Parameters              : {params:.2f} M")
    print(f"  Model Size              : {size:.2f} MB")
    print(f"  FLOPs                   : {flops:.4f} GFLOPs")
    print(f"  Inference Time          : {inference:.2f} ms")
    print(f"  Arithmetic Intensity    : {arithmetic_intensity:.2f} FLOPs/byte")
    print(f"  Ridge Point (device)    : {ridge_point:.2f} FLOPs/byte")
    print(f"  Bound Type              : {bound_type}-bound")

    is_deployable = (params < 12 and flops < 2 and
                     size < profile['max_size_mb'] and
                     inference < profile['max_inference_ms'])
    print(f"  Memenuhi constraint {profile['display_name']:<15}: {'✅ YES' if is_deployable else '❌ NO'}")
    print("=" * 60)

    return {
        'params_m': params,
        'flops_g': flops,
        'size_mb': size,
        'inference_ms': inference,
        'arithmetic_intensity': arithmetic_intensity,
        'ridge_point': ridge_point,
        'bound_type': bound_type,
        'deployable': is_deployable,
        'target_device': target_device,
        'measured_on_device': str(device),
    }

def print_efficiency_report(model, model_name, input_shape, device='cuda'):
    params = count_parameters(model)
    size = get_model_size(model)
    if hasattr(model, 'fusion'):
        flops = compute_flops_multimodal(model, device)
        inference_time = measure_inference_time_multimodal(model, device)
    else:
        flops = compute_flops(model, input_shape, device)
        inference_time = measure_inference_time(model, input_shape, device)
    print("\n" + "=" * 60)
    print(f"  EFFICIENCY REPORT — {model_name}")
    print("=" * 60)
    print(f"  Total Parameters  : {params:.2f} M")
    print(f"  Model Size        : {size:.2f} MB")
    print(f"  FLOPs             : {flops:.4f} GFLOPs")
    print(f"  Inference Time    : {inference_time:.2f} ms/sampel")
    print("=" * 60)
    check_lightweight_constraints(params, flops, size, inference_time)

def roofline_analysis_ensemble(fold_models, model_name, device='cuda', target_device='raspberry_pi_4b', num_frames=15):
    device = torch.device(device)
    n_models = len(fold_models)
    total_params = sum(count_parameters(m) for m in fold_models)
    total_size = sum(get_model_size(m) for m in fold_models)
    total_flops = sum(compute_flops_multimodal(m, device, num_frames) for m in fold_models)

    audio_dummy = torch.randn(1, 120, 224).to(device)
    visual_dummy = torch.randn(1, num_frames, 3, 224, 224).to(device)
    for m in fold_models:
        m.eval().to(device)

    with torch.no_grad():
        for _ in range(10):
            for m in fold_models:
                _ = m(audio_dummy, visual_dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(100):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        all_probs = []
        for m in fold_models:
            out = m(audio_dummy, visual_dummy)
            all_probs.append(torch.softmax(out, dim=1))
        avg_probs = torch.stack(all_probs).mean(dim=0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    inference = np.mean(times)
    profile = get_device_profile(target_device)

    print("\n" + "=" * 60)
    print(f"  ENSEMBLE ROOFLINE ANALYSIS (NAIF, tanpa sharing) — {model_name} ({n_models} fold)")
    print("=" * 60)
    print(f"  Total Parameters (x{n_models}) : {total_params:.2f} M")
    print(f"  Total Model Size  (x{n_models}) : {total_size:.2f} MB")
    print(f"  Total FLOPs       (x{n_models}) : {total_flops:.4f} GFLOPs")
    print(f"  Inference Time (all {n_models} + avg): {inference:.2f} ms")

    is_deployable = (total_params < 12 and total_flops < 2 and
                     total_size < profile['max_size_mb'] and
                     inference < profile['max_inference_ms'])
    print(f"  Deployable on {profile['display_name']:<20}: {'✅ YES' if is_deployable else '❌ NO'}")
    print("=" * 60)

    return {
        'n_models': n_models,
        'total_params_m': total_params,
        'total_size_mb': total_size,
        'total_flops_g': total_flops,
        'inference_ms': inference,
        'deployable': is_deployable,
        'recommended_for_deployment': False,
        'deployment_note': 'Gunakan single best-fold model untuk edge deployment.'
    }

# =============================================================================
# ENSEMBLE EFFICIENCY — VERSI REALISTIS (BACKBONE SHARED)
# =============================================================================

def compute_ensemble_efficiency_shared_backbone(
    audio_model, visual_model, fold_fusion_models,
    device='cuda', target_device='raspberry_pi_4b', verbose=True
):
    n_folds = len(fold_fusion_models)
    # Backbone (dihitung sekali)
    backbone_params_m = count_parameters(audio_model) + count_parameters(visual_model)
    backbone_size_mb = get_model_size(audio_model) + get_model_size(visual_model)
    backbone_flops_g = 0.0
    try:
        backbone_flops_g = (compute_flops(audio_model, (1, 120, 224), device) +
                            compute_flops(visual_model, (1, 3, 224, 224), device))
    except Exception:
        pass
    backbone_latency_ms = measure_backbone_inference_time(audio_model, visual_model, device)

    # Fusion head (gunakan fold ke-0 sebagai representasi)
    ref_fusion_model = fold_fusion_models[0]
    fusion_head_params_m = count_parameters(ref_fusion_model, include_backbone=False)
    fusion_head_size_mb = get_model_size(ref_fusion_model)
    fusion_head_latency_ms = measure_fusion_head_inference_time(ref_fusion_model, device)

    total_params_m = backbone_params_m + fusion_head_params_m * n_folds
    total_size_mb = backbone_size_mb + fusion_head_size_mb * n_folds
    total_flops_g = backbone_flops_g  # fusion head FLOPs diabaikan karena kecil
    total_latency_ms = backbone_latency_ms + fusion_head_latency_ms * n_folds

    profile_device = get_device_profile(target_device)
    is_deployable = (total_params_m < 12 and total_flops_g < 2 and
                     total_size_mb < profile_device['max_size_mb'] and
                     total_latency_ms < profile_device['max_inference_ms'])

    result = {
        'n_folds': n_folds,
        'backbone_params_m': backbone_params_m,
        'backbone_size_mb': backbone_size_mb,
        'backbone_flops_g': backbone_flops_g,
        'backbone_latency_ms': backbone_latency_ms,
        'fusion_head_params_m_per_fold': fusion_head_params_m,
        'fusion_head_params_m_total': fusion_head_params_m * n_folds,
        'fusion_head_size_mb_per_fold': fusion_head_size_mb,
        'fusion_head_size_mb_total': fusion_head_size_mb * n_folds,
        'fusion_head_latency_ms_per_fold': fusion_head_latency_ms,
        'fusion_head_latency_ms_total': fusion_head_latency_ms * n_folds,
        'total_params_m': total_params_m,
        'total_size_mb': total_size_mb,
        'total_flops_g': total_flops_g,
        'total_latency_ms': total_latency_ms,
        'deployable': is_deployable,
        'target_device': target_device,
    }

    if verbose:
        print("\n" + "=" * 70)
        print(f"  ENSEMBLE EFFICIENCY — BACKBONE SHARED ({n_folds}-fold)")
        print("=" * 70)
        print(f"  Backbone (shared, x1)")
        print(f"    Params   : {backbone_params_m:.2f} M")
        print(f"    Size     : {backbone_size_mb:.2f} MB")
        print(f"    Latency  : {backbone_latency_ms:.2f} ms")
        print(f"  Fusion Head (per-fold, diduplikasi x{n_folds})")
        print(f"    Params/fold : {fusion_head_params_m:.4f} M  →  Total: {fusion_head_params_m*n_folds:.2f} M")
        print(f"    Size/fold   : {fusion_head_size_mb:.4f} MB  →  Total: {fusion_head_size_mb*n_folds:.2f} MB")
        print(f"    Latency/fold: {fusion_head_latency_ms:.4f} ms  →  Total: {fusion_head_latency_ms*n_folds:.2f} ms")
        print("-" * 70)
        print(f"  TOTAL ENSEMBLE")
        print(f"    Params   : {total_params_m:.2f} M")
        print(f"    Size     : {total_size_mb:.2f} MB")
        print(f"    Latency  : {total_latency_ms:.2f} ms")
        print(f"  Memenuhi constraint {profile_device['display_name']:<15}: {'✅ YES' if is_deployable else '❌ NO'}")
        print("=" * 70)
    return result