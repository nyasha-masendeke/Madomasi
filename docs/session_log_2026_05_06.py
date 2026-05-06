"""Generate docs/session_log_2026_05_06.pdf from this session's work log."""
from fpdf import FPDF
from datetime import date

SESSION_DATE = "2026-05-06"
OUTPUT_PATH = "docs/session_log_2026_05_06.pdf"

SECTIONS = [
    {
        "title": "Session Overview",
        "body": (
            "Date: 2026-05-06\n"
            "Repository: nyasha-masendeke/Madomasi\n"
            "Branch: Madomasi (default branch)\n"
            "Working directory: C:\\Users\\PC\\Desktop\\Madomasi\n\n"
            "This session covered a full build-out of the Tomato AI Diagnostics application: "
            "backend/frontend integration, live webcam prediction, real-time training curves, "
            "two-stage transfer learning, dataset wiring, bug fixes, and project cleanup."
        ),
    },
    {
        "title": "1. Project Structure",
        "body": (
            "app.py               - Streamlit entry point\n"
            "main.py              - CLI entry point (train/evaluate/predict/convert)\n"
            "pipeline.py          - Core ML: load_datasets, build_model, train_model,\n"
            "                       train_two_stage, predict_image, evaluate_model,\n"
            "                       convert_model\n"
            "config.py            - DISEASE_CLASSES (10 classes), page config, CSS\n"
            "streamlit_callback.py- Keras callback -> real-time Streamlit charts\n"
            "components/\n"
            "  main_ui.py         - Inference tab + Training Dashboard tab\n"
            "  sidebar.py         - Confidence slider, disease filter, ROI, system stats\n"
            "src/\n"
            "  data/dataset.py    - 80/10/10 dataset splitter\n"
            "  utils/recommendations.py - Disease -> treatment text (all 10 classes)\n"
            "  utils/predictions.py     - Simulated inference helper\n"
            "  utils/training_utils.py  - DashboardCallback, get_sys_stats\n"
            "data/\n"
            "  raw/raw/tomato/    - 10,584 original images (10 class subfolders)\n"
            "  splits/            - train (10,532) / val (2,106) / test (2,081)\n"
            "models/trained/      - Saved Keras checkpoints\n"
            "models/tflite/       - TFLite exports for edge deployment\n"
            "outputs/             - Learning curve HTML files (saved after training)\n"
            "requirements-frontend.txt  - streamlit, plotly, pillow, pandas,\n"
            "                            streamlit-webrtc, av\n"
            "requirements-backend.txt   - tensorflow, keras, opencv-headless,\n"
            "                            scikit-learn, psutil, numpy\n"
        ),
    },
    {
        "title": "2. Backend / Frontend Integration",
        "body": (
            "Problem: main_ui.py had two run_app() definitions, ~500 lines of dead "
            "triple-quoted code, a key mismatch (params['conf'] vs sidebar returning "
            "'confidence'), an undefined 'cb' variable, and a broken progress_bar.progress() "
            "call signature.\n\n"
            "pipeline.py was an older version using MobileNetV2 and the legacy "
            "ImageDataGenerator API with a hardcoded wrong DATA_DIR.\n\n"
            "Fixes applied:\n"
            "- Rewrote main_ui.py: single clean run_app(), fixed all key mismatches,\n"
            "  fixed progress bar, wired get_recommendation() from utils.\n"
            "- Rewrote pipeline.py: MobileNetV3Small + image_dataset_from_directory,\n"
            "  sparse_categorical_crossentropy, added evaluate_model() and convert_model().\n"
            "- Removed st.rerun() from StreamlitTrainCallback.on_epoch_end() -- it was\n"
            "  restarting the Streamlit script and killing model.fit() mid-training.\n"
            "- Filled in CLI stubs in main.py (evaluate, predict, convert).\n"
            "- Fixed CSS typo: ##46499e -> #46499e."
        ),
    },
    {
        "title": "3. Live Webcam Frame-Level Prediction",
        "body": (
            "Added FramePredictor class to components/main_ui.py:\n\n"
            "- Callable for streamlit_webrtc's video_frame_callback\n"
            "- Loads the Keras model lazily on first frame (avoids blocking UI)\n"
            "- Runs MobileNetV3 inference every 20 frames (~1.5x/second at 30fps)\n"
            "- Draws diagnosis overlay (disease name + confidence %) on each frame\n"
            "  using OpenCV cv2.putText and cv2.rectangle\n"
            "- Thread-safe: threading.Lock guards model access (callback runs in\n"
            "  a background thread separate from the Streamlit main thread)\n"
            "- Stored in session_state so model survives Streamlit reruns\n"
            "- Shows last prediction as metric cards + treatment recommendation\n"
            "  below the video stream\n\n"
            "Color coding on overlay:\n"
            "  Green  = Healthy\n"
            "  Red    = Diseased (above confidence threshold)\n"
            "  Orange = Below confidence threshold\n\n"
            "Packages added: streamlit-webrtc>=0.47.0, av>=10.0.0\n\n"
            "Camera permission note: NotAllowedError: Permission denied is a browser\n"
            "security restriction, not a code bug. Requires user to click Allow on\n"
            "the browser permission popup. HTTPS required for non-localhost URLs."
        ),
    },
    {
        "title": "4. Real-Time Learning Curves",
        "body": (
            "Updated StreamlitTrainCallback to accept a chart_placeholder parameter.\n"
            "On every epoch end:\n\n"
            "  1. Progress bar updates: 'Epoch N / Total'\n"
            "  2. Four metric cards: Train Acc | Train Loss | Val Acc | Val Loss\n"
            "  3. Plotly 2-subplot chart redraws in place:\n"
            "       Left:  Accuracy (train=green, val=orange-dashed)\n"
            "       Right: Loss     (train=red, val=blue-dashed)\n\n"
            "After training completes, _save_learning_curves() writes an interactive\n"
            "standalone HTML file to outputs/learning_curves_<timestamp>.html\n"
            "using fig.write_html().\n\n"
            "Fixed default Dataset Directory in UI: 'data/train' -> 'data/splits/train'."
        ),
    },
    {
        "title": "5. Dataset Wiring",
        "body": (
            "Dataset location: data/raw/raw/tomato/ (double 'raw' in path)\n"
            "10 class folders, ~1,100 images each (total: 10,584 images)\n\n"
            "Classes (sorted alphabetically = Keras index order):\n"
            "  0  Tomato___Bacterial_spot\n"
            "  1  Tomato___Early_blight\n"
            "  2  Tomato___healthy\n"
            "  3  Tomato___Late_blight\n"
            "  4  Tomato___Leaf_Mold\n"
            "  5  Tomato___Septoria_leaf_spot\n"
            "  6  Tomato___Spider_mites Two-spotted_spider_mite\n"
            "  7  Tomato___Target_Spot\n"
            "  8  Tomato___Tomato_mosaic_virus\n"
            "  9  Tomato___Tomato_Yellow_Leaf_Curl_Virus\n\n"
            "Split (80/10/10 via src/data/dataset.py):\n"
            "  Train: 10,532  |  Val: 2,106  |  Test: 2,081\n\n"
            "Pipeline verification: load_datasets() + build_model() tested end-to-end.\n"
            "Model output shape confirmed: (None, 10)\n\n"
            "Bug fixed: class_names attribute lost after .prefetch() call.\n"
            "Fix: cache class_names before prefetch and re-attach after."
        ),
    },
    {
        "title": "6. Two-Stage Transfer Learning",
        "body": (
            "Added train_two_stage() to pipeline.py:\n\n"
            "Stage 1 - Feature Extraction:\n"
            "  Base model: MobileNetV3Small FROZEN (ImageNet weights preserved)\n"
            "  Trains: classification head only (GlobalAvgPool -> Dropout -> Dense(10))\n"
            "  Learning rate: 0.001\n"
            "  Saves best checkpoint by val_accuracy\n\n"
            "Stage 2 - Fine-Tuning:\n"
            "  Reloads best Stage 1 checkpoint\n"
            "  Base model: UNFROZEN (entire network trains end-to-end)\n"
            "  Learning rate: 0.0001 (10x lower to avoid destroying ImageNet features)\n"
            "  Saves best checkpoint by val_accuracy (overwrites if better)\n\n"
            "Both stages use:\n"
            "  ModelCheckpoint (save_best_only=True, monitor=val_accuracy)\n"
            "  EarlyStopping (patience=3, monitor=val_loss, restore_best_weights=True)\n\n"
            "UI: Training Dashboard now has a 'Two-Stage Training' toggle (default ON)\n"
            "with separate epoch/LR controls for each stage. Falls back to single-stage\n"
            "train_model() when toggled OFF."
        ),
    },
    {
        "title": "7. Bug Fixes",
        "body": (
            "StreamlitAPIException - multiselect default mismatch:\n"
            "  sidebar.py had hardcoded default=['Early Blight', 'Late Blight', ...]\n"
            "  after DISEASE_CLASSES was updated to Tomato___* folder names.\n"
            "  Fix: changed default=DISEASE_CLASSES (all classes selected by default).\n\n"
            "Stale 'Healthy' string comparisons:\n"
            "  main_ui.py and sidebar.py used == 'Healthy' in multiple places.\n"
            "  Fix: changed to 'healthy' in disease.lower() to match 'Tomato___healthy'.\n\n"
            "recommendations.py stale keys:\n"
            "  Dict keys used old display names ('Early Blight', etc.).\n"
            "  Fix: rewrote all keys to match actual folder names, added all 10 classes.\n\n"
            "dataset.py UnicodeEncodeError:\n"
            "  print() with emoji caused cp1252 encoding error on Windows.\n"
            "  Fix: removed emoji from print statement.\n\n"
            "dataset.py wrong raw path:\n"
            "  Hardcoded 'data/raw/tomato' -- actual path is 'data/raw/raw/tomato'.\n"
            "  Fix: updated path in split_dataset() call."
        ),
    },
    {
        "title": "8. Project Cleanup",
        "body": (
            "Deleted 4 duplicate 'Copy' files (217 lines of dead code removed):\n"
            "  - config - Copy.py\n"
            "  - main - Copy.py\n"
            "  - pipeline - Copy.py\n"
            "  - streamlit_callback - Copy.py\n\n"
            ".gitignore improvements:\n"
            "  - models/**/*.keras, *.tflite, *.h5 (catch models in any subdirectory)\n"
            "  - outputs/*.html, outputs/*.png (prevent learning curve files from committing)\n\n"
            "README.md added with:\n"
            "  - Feature overview\n"
            "  - Full disease class table\n"
            "  - Project structure tree\n"
            "  - Quick start guide\n"
            "  - CLI training commands\n"
            "  - Two-stage training explanation\n"
            "  - Edge deployment (Raspberry Pi 5) instructions\n"
            "  - Webcam permission notes\n"
            "  - Requirements breakdown"
        ),
    },
    {
        "title": "9. Commit History",
        "body": (
            "c0aa4b3  Remove Copy files and tighten .gitignore\n"
            "f241837  Add README.md\n"
            "7612eb4  Add camera permission guidance for webcam tab\n"
            "d199e8d  Fix stale class name references after DISEASE_CLASSES update\n"
            "b5c2825  Add two-stage feature extraction and fine-tuning pipeline\n"
            "79c782d  Add real-time learning curves during training + save to outputs/\n"
            "719826e  Wire dataset and fix pipeline for 10-class tomato dataset\n"
            "71ce3d0  Add live webcam frame-level disease prediction\n"
            "caf9328  Resolve merge conflicts: keep fixed versions from claude branch\n"
            "e1efba1  Merge claude/naughty-benz-c78141: fix backend/frontend integration\n"
            "8befc6e  Fix backend/frontend integration: rewrite main_ui.py and pipeline.py"
        ),
    },
    {
        "title": "10. Outstanding Professional Suggestions",
        "body": (
            "High priority:\n"
            "  - Save class_names inside model metadata at training time to eliminate\n"
            "    manual DISEASE_CLASSES sync requirement\n"
            "  - Add data augmentation to load_datasets() (flips, rotation, brightness)\n"
            "  - Add display name mapping for class folder names in the UI\n\n"
            "Medium priority:\n"
            "  - Add per-class precision/recall/F1 to evaluate_model()\n"
            "  - Add confidence calibration (temperature scaling) post-training\n"
            "  - Add model status indicator in UI (green/red: model file exists or not)\n"
            "  - Webcam scans should increment session statistics footer\n\n"
            "Lower priority:\n"
            "  - Add FastAPI layer for serving predictions to other clients\n"
            "  - Add model versioning registry (models/registry.json)\n"
            "  - Centralise all path constants in config.py or .env\n"
            "  - Write real smoke tests in tests/test_smoke.py\n"
        ),
    },
]


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(255, 75, 75)
        self.cell(0, 8, "Tomato AI Diagnostics - Session Log", align="L")
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, SESSION_DATE, align="R")
        self.ln(2)
        self.set_draw_color(255, 75, 75)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Page {self.page_no()} | nyasha-masendeke/Madomasi", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(70, 73, 158)
        self.set_fill_color(240, 240, 250)
        self.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def section_body(self, body):
        self.set_font("Courier", size=8)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, body)
        self.ln(4)


def generate():
    pdf = PDF()
    pdf.set_margins(14, 16, 14)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    # Cover title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 75, 75)
    pdf.cell(0, 12, "Tomato AI Diagnostics", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(70, 73, 158)
    pdf.cell(0, 8, "Development Session Log", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, SESSION_DATE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    for section in SECTIONS:
        pdf.section_title(section["title"])
        pdf.section_body(section["body"])

    pdf.output(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
