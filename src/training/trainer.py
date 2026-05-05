"""Model training pipeline (feature extraction + fine-tuning)."""
def build_model(base_name: str, input_shape: tuple, freeze_base: bool = True):
    """Construct transfer learning model."""
    # TODO: Load base model, add classification head
    pass

def train_step(model, dataloader, epochs: int, lr: float):
    """Run training loop."""
    # TODO: Implement training logic
    pass
