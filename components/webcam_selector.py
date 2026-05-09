"""
webcam_selector — custom Streamlit component with camera device selection.

Serves the frontend HTML from Streamlit's own static file server
(localhost:8501/app/static/webcam.html) so the iframe shares the same origin
as the Streamlit app — this lets getUserMedia inherit the camera permission
that was already granted to the Streamlit origin.

Requires:  .streamlit/config.toml  →  [server] enableStaticServing = true
           static/webcam.html      →  the camera widget frontend
"""
import time
import streamlit.components.v1 as _stc

# Hardcode port 8501 — the app always runs on this port.
# Append a version stamp so browsers never serve a stale cached iframe.
_VER = int(time.time() // 3600)          # changes once per hour at most
_STATIC_URL = f"http://localhost:8501/app/static/webcam.html?v={_VER}"


def webcam_selector(height: int = 420, key: str = "webcam_selector"):
    """
    Renders a camera-selection widget with a real-time bounding-box overlay.
    Returns ``"data:image/jpeg;base64,..."`` when the user clicks Take Photo,
    or ``None`` otherwise.
    """
    _func = _stc.declare_component(
        "webcam_selector",
        url=_STATIC_URL,
    )
    return _func(key=key, default=None, height=height)
