"""Basic smoke tests for project setup."""
def test_imports():
    import config
    import src.data.dataset
    import src.inference.predictor
    assert config.DISEASE_CLASSES is not None

def test_cli_structure():
    from main import main
    # Verify argparse structure exists
    assert callable(main)
