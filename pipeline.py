"""Backwards-compatibility facade.

All logic now lives in the src/ package.  Existing imports of the form
    from pipeline import predict_image, train_head, ...
continue to work without modification.
"""
from src.training.data import (        # noqa: F401
    next_training_output_dir as _next_training_output_dir,
    load_datasets,
    build_model,
    split_dataset,
    check_class_balance,
)
from src.training.features import (    # noqa: F401
    extract_features,
    train_head,
    fine_tune_model,
    train_model,
    train_two_stage,
)
from src.inference.predict import (    # noqa: F401
    predict_image,
    log_inference,
)
from src.inference.gradcam import (    # noqa: F401
    compute_gradcam,
)
from src.evaluation.evaluate import (  # noqa: F401
    evaluate_model,
    compute_tsne,
)
from src.export.tflite import (        # noqa: F401
    convert_model,
)
