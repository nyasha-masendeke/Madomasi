"""Shared UI utility functions used across multiple tab components."""
import re
from datetime import datetime
from pathlib import Path

import streamlit as st


def html(raw: str) -> None:
    """Render HTML via st.markdown, collapsing blank lines so the markdown
    parser never restarts paragraph mode inside an HTML block."""
    compact = re.sub(r'\n\s*\n', '\n', raw).strip()
    st.markdown(compact, unsafe_allow_html=True)


def find_models() -> list[str]:
    base = Path("models/trained")
    if not base.exists():
        return []
    return sorted(
        [str(p) for p in base.rglob("*.keras")],
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )


def detect_dir(candidates: list[str], fallback: str = "") -> str:
    for p in candidates:
        if p and Path(p).exists():
            return p
    return fallback


def model_picker(label: str, key: str, default: str = "models/trained/latest.keras") -> str:
    available = find_models()
    CUSTOM = "Custom path..."
    if available:
        choice = st.selectbox(
            label,
            options=available + [CUSTOM],
            format_func=lambda x: Path(x).name if x != CUSTOM else CUSTOM,
            key=f"{key}_select",
        )
        if choice == CUSTOM:
            return st.text_input(
                "Custom model path", "", key=f"{key}_custom",
                placeholder="models/trained/run_xxx/stage2_ft.keras",
            )
        return choice
    detected = detect_dir([default, "models/trained/latest.keras"], fallback="")
    return st.text_input(
        label, detected,
        placeholder="models/trained/run_xxx/stage2_ft.keras",
        key=key,
    )


def model_info_html(path: str) -> str:
    if not path or not Path(path).exists():
        return ""
    p       = Path(path)
    size_kb = p.stat().st_size / 1024
    mtime   = datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d %Y")
    size_str = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
    return (
        f'<span style="font-size:0.72rem;color:#374151;">'
        f'&#128196; {p.name} &nbsp;·&nbsp; {size_str} &nbsp;·&nbsp; {mtime}</span>'
    )


def model_status_html(path: str) -> str:
    exists = bool(path) and Path(path).exists()
    dot    = "&#9679;"
    if exists:
        return f'<span class="model-status-ok">{dot} Model Ready</span>'
    return f'<span class="model-status-missing">{dot} Model Not Found</span>'


def load_latest_history() -> dict:
    import json
    base = Path("outputs")
    if not base.exists():
        return {}
    dirs = sorted(
        [p for p in base.iterdir()
         if p.is_dir() and p.name.startswith("Training") and p.name[8:].isdigit()],
        key=lambda p: int(p.name[8:]),
        reverse=True,
    )
    for td in dirs:
        files = sorted(td.glob("history_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            try:
                return json.loads(files[0].read_text())
            except Exception:
                continue
    return {}
