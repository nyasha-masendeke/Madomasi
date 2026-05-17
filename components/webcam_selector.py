"""
webcam_selector — custom Streamlit component with camera device selection.

Uses declare_component(path=) so Streamlit serves the frontend files itself.
No dev server or static-serving config required.
"""
from pathlib import Path
import streamlit.components.v1 as _stc

_FRONTEND = str(Path(__file__).parent / "webcam_frontend")
_component = _stc.declare_component("webcam_selector", path=_FRONTEND)


def webcam_selector(height: int = 420, key: str = "webcam_selector"):
    """
    Renders a camera-selection widget with a real-time bounding-box overlay.
    Returns ``"data:image/jpeg;base64,..."`` when the user clicks Take Photo,
    or ``None`` otherwise.
    """
    return _component(key=key, default=None, height=height)
