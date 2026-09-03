# -*- coding: utf-8 -*-
"""
evaluation/__init__.py
======================
Package untuk evaluasi metrik dan visualisasi.
"""

from .metrics import (
    compute_classification_metrics,
    evaluate_test_set,
    evaluate_unimodal_model,
    evaluate_multimodal_model,
    evaluate_multimodal_with_masking,
    save_masking_evaluation,
    convert_to_serializable,
    save_test_metrics,
    geometric_mean_fusion,
    evaluate_geometric_mean,
)
from .efficiency import (
    count_parameters,
    get_model_size,
    compute_flops,
    measure_inference_time,
    print_efficiency_report,
    check_lightweight_constraints,
    compute_flops_multimodal,
    measure_inference_time_multimodal,
    roofline_analysis,
    roofline_analysis_ensemble,
    get_device_profile,
    DEVICE_PROFILES,
    get_checkpoint_size_from_file,
    compute_ensemble_efficiency_shared_backbone,
)
from .benchmark import (
    compare_experiments,
    generate_comparison_table,
    save_comparison_table,
    print_comparison_table,
    load_experiment_metrics,
    evaluate_experiment_efficiency,
)
from .visualize import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_accuracy_comparison,
    plot_efficiency_comparison,
    plot_f1_comparison,
)

try:
    from .metrics import (
        evaluate_with_tta,
        evaluate_with_knn_reranking,
        apply_temperature_scaling,
        knn_re_ranking,
    )
except ImportError:
    def evaluate_with_tta(*args, **kwargs):
        raise NotImplementedError("evaluate_with_tta tidak tersedia")

    def evaluate_with_knn_reranking(*args, **kwargs):
        raise NotImplementedError("evaluate_with_knn_reranking tidak tersedia")

    def apply_temperature_scaling(*args, **kwargs):
        raise NotImplementedError("apply_temperature_scaling tidak tersedia")

    def knn_re_ranking(*args, **kwargs):
        raise NotImplementedError("knn_re_ranking tidak tersedia")


__all__ = [
    'compute_classification_metrics',
    'evaluate_test_set',
    'evaluate_unimodal_model',
    'evaluate_multimodal_model',
    'evaluate_multimodal_with_masking',
    'save_masking_evaluation',
    'convert_to_serializable',
    'save_test_metrics',
    'geometric_mean_fusion',
    'evaluate_geometric_mean',
    'evaluate_with_tta',
    'evaluate_with_knn_reranking',
    'apply_temperature_scaling',
    'knn_re_ranking',
    'count_parameters',
    'get_model_size',
    'compute_flops',
    'measure_inference_time',
    'print_efficiency_report',
    'check_lightweight_constraints',
    'compute_flops_multimodal',
    'measure_inference_time_multimodal',
    'roofline_analysis',
    'roofline_analysis_ensemble',
    'get_device_profile',
    'DEVICE_PROFILES',
    'get_checkpoint_size_from_file',
    'compute_ensemble_efficiency_shared_backbone',
    'compare_experiments',
    'generate_comparison_table',
    'save_comparison_table',
    'print_comparison_table',
    'load_experiment_metrics',
    'evaluate_experiment_efficiency',
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_accuracy_comparison',
    'plot_efficiency_comparison',
    'plot_f1_comparison',
]