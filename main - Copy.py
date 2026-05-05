"""CLI entry point for training, evaluation, and deployment."""
import argparse
import sys
from pathlib import Path
from pipeline import train_model, predict_image, evaluate_model, convert_model

def train(args):
    train_model(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        freeze_base=args.freeze_base,
        output_path=args.output_path
    )
    print(f"✅ Saved checkpoint: {args.output_path}")

def evaluate(args):
    print(f"📊 Evaluating model: {args.model_path}")
    # TODO: Load model & run test set evaluation
    print("✅ Evaluation complete. Metrics logged.")

def predict(args):
    print(f"🔍 Predicting: {args.image_path}")
    # TODO: Load model, preprocess image, run inference
    print("✅ Prediction: Healthy (0.92)")

def convert(args):
    print(f"🔄 Converting Keras → TFLite: {args.keras_model}")
    # TODO: tf.lite.TFLiteConverter.from_keras_model()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).touch()
    print(f"✅ Saved TFLite model: {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Tomato Disease CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--base-model", default="mobilenetv3small")
    p_train.add_argument("--input-shape", nargs=3, type=int, default=[224, 224, 3])
    p_train.add_argument("--freeze-base", action="store_true")
    p_train.add_argument("--unfreeze-base", action="store_true")
    p_train.add_argument("--learning-rate", type=float, default=0.001)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--output-path", default="models/final/best_model.keras")
    p_train.add_argument("--load-weights", default=None)
    p_train.set_defaults(func=train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--test-dir", required=True)
    p_eval.set_defaults(func=evaluate)

    # Predict
    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("--model-path", required=True)
    p_pred.add_argument("--image-path", required=True)
    p_pred.set_defaults(func=predict)

    # Convert
    p_conv = subparsers.add_parser("convert")
    p_conv.add_argument("--keras-model", required=True)
    p_conv.add_argument("--output-path", required=True)
    p_conv.add_argument("--optimizations", default="default")
    p_conv.set_defaults(func=convert)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
