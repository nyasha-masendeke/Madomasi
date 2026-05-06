"""CLI entry point for training, evaluation, and deployment."""
import argparse
import sys
from pipeline import train_model, predict_image, evaluate_model, convert_model


def train(args):
    _, saved_path = train_model(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        freeze_base=args.freeze_base,
        output_path=args.output_path,
    )
    print(f"Saved checkpoint: {saved_path}")


def evaluate(args):
    print(f"Evaluating model: {args.model_path}")
    metrics = evaluate_model(args.model_path, args.test_dir)
    print(f"Loss     : {metrics['loss']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")


def predict(args):
    print(f"Predicting: {args.image_path}")
    result = predict_image(args.model_path, args.image_path)
    print(f"Disease    : {result['disease']}")
    print(f"Confidence : {result['confidence']:.1%}")
    if not result["passes_threshold"]:
        print("Warning: confidence below threshold")


def convert(args):
    print(f"Converting: {args.keras_model} -> {args.output_path}")
    output = convert_model(args.keras_model, args.output_path)
    print(f"Saved TFLite model: {output}")


def main():
    parser = argparse.ArgumentParser(description="Tomato Disease CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--base-model", default="mobilenetv3small")
    p_train.add_argument("--freeze-base", action="store_true")
    p_train.add_argument("--learning-rate", type=float, default=0.001)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--output-path", default="models/trained/latest.keras")
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
    p_conv.set_defaults(func=convert)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
