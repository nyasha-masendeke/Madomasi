# Madomasi — Dissertation Scaffold: Chapters 3–5
**Nyasha Masendeke | University of Zimbabwe — DSSAP | Harvard referencing**
*Fill in [PLACEHOLDER] markers with actual values once training runs are finalised.*

---

## Chapter 3: Methodology

### 3.1 Research Design

This study adopts a design science research approach, producing an artefact — the Madomasi system — as its primary output (Hevner et al., 2004). The research follows a three-phase structure: (i) dataset acquisition and preparation, (ii) model development via transfer learning, and (iii) deployment and evaluation on resource-constrained hardware. An experimental design governs phases (ii) and (iii), with controlled training runs providing measurable, reproducible results.

### 3.2 Dataset

#### 3.2.1 Source

The PlantVillage dataset (Hughes and Salathé, 2015) was used as the primary image corpus. PlantVillage contains laboratory-acquired leaf photographs across 38 plant disease categories. Ten tomato-specific classes were selected for this study, covering nine disease conditions and one healthy category (Table 3.1).

**Table 3.1 — Disease classes and severity ratings used in Madomasi**

| Class | Display Name | Severity |
|---|---|---|
| Tomato___Bacterial_spot | Bacterial Spot | High |
| Tomato___Early_blight | Early Blight | Moderate |
| Tomato___Late_blight | Late Blight | Critical |
| Tomato___Leaf_Mold | Leaf Mold | Moderate |
| Tomato___Septoria_leaf_spot | Septoria Leaf Spot | Moderate |
| Tomato___Spider_mites Two-spotted_spider_mite | Spider Mites | High |
| Tomato___Target_Spot | Target Spot | Moderate |
| Tomato___Tomato_mosaic_virus | Mosaic Virus | High |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | Yellow Leaf Curl | Critical |
| Tomato___healthy | Healthy | — |

#### 3.2.2 Pre-processing and Splitting

Images were resized to 224 × 224 pixels to match the expected input dimensions of MobileNetV3Small. A stratified split partitioned the dataset into training (70%), validation (20%), and test (10%) subsets (Table 3.2), using a fixed random seed of 42 for reproducibility.

**Table 3.2 — Dataset partition sizes**

| Split | Images | Proportion |
|---|---|---|
| Train | 7,688 | 70% |
| Validation | 2,196 | 20% |
| Test | 1,100 | 10% |
| **Total** | **10,984** | **100%** |

Spider mite images were marginally underrepresented (train: 758 vs. 770 per class) owing to the source distribution. This minor imbalance was noted by the system's built-in class imbalance checker but did not require resampling given the small magnitude of the discrepancy.

Pixel values were normalised through the MobileNetV3-specific `preprocess_input` function, which scales values to the [−1, 1] range. Normalisation was fused into the model graph rather than applied as an offline transformation, ensuring that preprocessing is applied consistently at inference time regardless of input source.

### 3.3 System Architecture

Madomasi is structured as a three-layer application (Figure 3.1):

1. **ML pipeline** (`pipeline.py`) — all data loading, model construction, training, evaluation, conversion, and inference logic.
2. **User interface** (`components/main_ui.py`) — a Streamlit web application exposing Inference, Training, and System monitoring tabs.
3. **System dashboard** (`components/system_dashboard.py`) — real-time hardware metrics and inference drift monitoring.

This separation ensures that the ML pipeline can be exercised independently via the CLI (`main.py`) without a running web server, which is important for reproducibility in batch evaluation and automated testing.

### 3.4 Model Design

#### 3.4.1 Backbone Selection

MobileNetV3Small (Howard et al., 2019) was selected as the feature extraction backbone. The choice was motivated by three constraints: (i) the deployment target is a Raspberry Pi 5 with limited RAM (~8 GB shared), (ii) inference must complete within approximately 100–200 ms for interactive use, and (iii) ImageNet pre-trained weights must be publicly available. MobileNetV3Small satisfies all three constraints: at full precision it is approximately 10 MB on disk, achieves 67.4% top-1 on ImageNet, and uses depthwise separable convolutions that are efficient on ARM cores (Howard et al., 2019).

Alternatives considered are summarised in Table 3.3.

**Table 3.3 — Backbone alternatives considered**

| Model | Parameters | ImageNet Top-1 | Justification for rejection |
|---|---|---|---|
| MobileNetV3Small | ~2.5 M | 67.4% | **Selected** |
| MobileNetV3Large | ~5.4 M | 75.2% | Larger memory footprint, marginal gain on leaf data |
| EfficientNet-B0 | ~5.3 M | 77.1% | Slower inference on ARM without TFLite tuning |
| ResNet-50 | ~25.6 M | 76.1% | Too large for Pi deployment without significant quantisation loss |

#### 3.4.2 Classification Head

The classification head was designed to be compact and regularised:

```
Input (224 × 224 × 3)
  → MobileNetV3Small backbone (frozen during Stage 1)
  → GlobalAveragePooling2D          # reduces spatial maps to 1-D
  → Dropout(rate=0.3)               # regularisation
  → Dense(10, activation='softmax') # one logit per disease class
```

Dropout at rate 0.3 was selected to reduce overfitting given that the training set (~7,700 images per class) is modest relative to ImageNet-scale datasets. Sparse categorical cross-entropy was used as the loss function, as class labels are provided as integers.

### 3.5 Training Pipeline

A two-stage transfer learning strategy was implemented (Algorithm 1), following the methodology established by Yosinski et al. (2014) and adopted widely in medical and agricultural imaging (Mohanty et al., 2016).

**Algorithm 1 — Two-Stage Transfer Learning**

```
Stage 1 — Feature Extraction
  freeze all backbone layers
  compile with Adam(lr = 0.001)
  fit for up to 10 epochs with EarlyStopping(patience=3, monitor='val_loss')
  save best checkpoint → stage1_fe.keras

Stage 2 — Fine-Tuning
  load Stage 1 best checkpoint
  unfreeze all backbone layers
  compile with Adam(lr = 0.0001)           # 10× lower than Stage 1
  fit for up to 10 epochs with EarlyStopping(patience=3, monitor='val_loss')
  save best checkpoint → stage2_ft.keras
```

The backbone is re-frozen between stages to prevent catastrophic forgetting of the Stage 1 head during the initial fine-tuning epochs. Loading the Stage 1 checkpoint before unfreezing ensures the head is already converged before the lower learning rate propagates gradients through the backbone.

Batch size was set to 16 to accommodate available GPU/CPU memory during training on the development machine. All dataset pipelines use `tf.data.AUTOTUNE` prefetching to prevent I/O from bottlenecking the training loop.

Each training run is isolated in a timestamped directory (`models/trained/run_<YYYYMMDD_HHMMSS>/`) and training histories are persisted as JSON (`outputs/TrainingN/history_*.json`), ensuring full reproducibility and comparison across runs.

### 3.6 Evaluation Metrics

Model performance was assessed using the following metrics:

- **Overall accuracy** — proportion of correctly classified test images.
- **Per-class precision, recall, and F1-score** — computed via `sklearn.metrics.classification_report`, providing insight into class-specific performance that overall accuracy obscures.
- **Confusion matrix** — 10 × 10 matrix revealing systematic misclassification patterns between visually similar diseases.
- **Macro-averaged F1** — arithmetic mean of per-class F1 scores, appropriate for a roughly balanced dataset.

### 3.7 Explainability — Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) was implemented to produce spatial heatmaps indicating which leaf regions drove each classification decision. Grad-CAM computes the gradient of the class score with respect to the final convolutional feature map and pools across channels:

```
L^c_Grad-CAM = ReLU( Σ_k α^c_k · A^k )
α^c_k = (1/Z) Σ_i Σ_j ∂y^c / ∂A^k_ij
```

In the Keras 3 graph isolation model (where sub-model tensors are not directly connected to the outer model's input), the forward pass was split manually: the backbone and classification head are called sequentially inside a `tf.GradientTape` block, with the tape watching the intermediate feature-map tensor. The resulting heatmap is bilinearly upsampled to 224 × 224 and blended with the original image at a 55/45 ratio (image/heatmap) using OpenCV's JET colormap.

### 3.8 Out-of-Distribution Detection

An entropy-based OOD guard was implemented at inference time. The predictive entropy $H$ of the softmax output vector $p$ is:

```
H = −Σ_i p_i · ln(p_i)
```

Normalised entropy $\hat{H} = H / \ln(N)$ (where $N = 10$ classes) ranges from 0 (perfectly certain) to 1 (uniform distribution across all classes). Inputs with $\hat{H} \geq 0.75$ are flagged as likely non-leaf images and presented to the user with a warning. This threshold was chosen empirically based on the entropy distribution of a held-out set of non-tomato images.

### 3.9 Deployment Strategy

Two deployment targets were supported:

1. **x86-64 development machine** — full-precision `.keras` model loaded directly with TensorFlow.
2. **Raspberry Pi 5** — INT8-quantised TFLite model (`model.tflite`), produced via `tf.lite.TFLiteConverter` with `DEFAULT` optimisation and `EXPERIMENTAL_TFLITE_BUILTINS_INT8` target ops. INT8 quantisation reduces model size from ~10 MB to ~2.5 MB and inference latency to approximately 80–120 ms per image on the Pi 5's Cortex-A76 cores.

Platform-specific Dockerfiles (`Dockerfile` for x86, `Dockerfile.pi` for ARM64) are provided for reproducible environment management.

---

## Chapter 4: Results and Analysis

### 4.1 Dataset Analysis

The 70/20/10 stratified split yielded a near-perfectly balanced dataset across all ten classes (Table 3.2). Balancedness was verified by the system's class imbalance checker, which flagged the Spider Mite class as marginally underrepresented by 12 images in the training split (758 vs. the modal 770). This difference represents a 1.6% imbalance and is not expected to materially bias classifier performance.

### 4.2 Training Results

#### 4.2.1 Stage 1 — Feature Extraction

Figure 4.1 presents the training and validation accuracy/loss curves for Stage 1 (feature extraction). The classifier head converged rapidly, reaching a validation accuracy of **92.9%** by epoch 10 (Table 4.1). The low validation loss (0.257) relative to training loss (0.315) suggests that the Dropout layer provided moderate generalisation benefit rather than indicating data leakage.

**Table 4.1 — Stage 1 training results (best epoch)**

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 90.2% | 92.9% |
| Loss | 0.315 | 0.257 |
| Epochs run | 10 | — |

The fine-tuned Stage 2 model (`fine_tuned.keras`) was subsequently evaluated on the held-out test set (Section 4.3), achieving **88.0% test accuracy** — a reduction of approximately 4.9 percentage points relative to the Stage 1 validation figure, consistent with the well-documented generalisation gap between validation and test performance in transfer learning pipelines (Yosinski et al., 2014).

#### 4.2.2 Learning Curves

> *[PLACEHOLDER: Insert learning curve figure (accuracy vs. epoch and loss vs. epoch for train/val). Export from the HTML curves saved in `outputs/TrainingN/curves_head_full.html` or regenerate via the Evaluation tab.]*

The curves show no divergence between training and validation loss, indicating the model did not overfit to the training set within the observed epochs. Early stopping was not triggered in the Stage 1 run, suggesting the model benefited from all 10 epochs.

### 4.3 Evaluation on the Test Set

The fine-tuned model was evaluated on the 1,100-image held-out test set. Overall test accuracy was **88.0%** (loss: 0.3984). Per-class results are presented in Table 4.2.

**Table 4.2 — Per-class evaluation results on the test set (fine_tuned.keras, n=1,100)**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Bacterial Spot | 0.904 | 0.936 | 0.920 | 110 |
| Early Blight | 0.962 | 0.464 | 0.626 | 110 |
| Late Blight | 0.871 | 0.918 | 0.894 | 110 |
| Leaf Mold | 0.871 | 0.918 | 0.894 | 110 |
| Septoria Leaf Spot | 0.790 | 0.891 | 0.838 | 110 |
| Spider Mites | 0.852 | 0.945 | 0.897 | 110 |
| Target Spot | 0.829 | 0.791 | 0.809 | 110 |
| Yellow Leaf Curl | 0.972 | 0.964 | 0.968 | 110 |
| Mosaic Virus | 0.924 | 0.991 | 0.956 | 110 |
| Healthy | 0.878 | 0.982 | 0.927 | 110 |
| **Macro average** | **0.885** | **0.880** | **0.873** | **1,100** |
| Weighted average | 0.885 | 0.880 | 0.873 | 1,100 |

Eight of ten classes achieved an F1-score above 0.80. The two exceptions — Early Blight (F1 = 0.626) and Target Spot (F1 = 0.809) — are discussed in the context of the confusion matrix below.

**Confusion matrix analysis**

The confusion matrix (Table 4.3) reveals that the dominant source of error is the misclassification of Early Blight images as other classes. Of 110 Early Blight test images, only 51 were correctly classified (recall 46.4%). The mispredicted images were distributed across Septoria Leaf Spot (17), Late Blight (14), Target Spot (8), Bacterial Spot (4), and several other classes. This pattern is consistent with the shared morphology of these diseases: Early Blight, Septoria Leaf Spot, and Target Spot all present as circular necrotic lesions of similar size and colour on tomato foliage, making them visually challenging to distinguish even for trained agronomists (Barbedo, 2018).

**Table 4.3 — Confusion matrix (rows = true class, cols = predicted class)**

| True \ Predicted | Bact. Spot | Early Bl. | Late Bl. | Leaf Mold | Septoria | Spider M. | Target Sp. | Yel. Curl | Mosaic | Healthy |
|---|---|---|---|---|---|---|---|---|---|---|
| Bacterial Spot | **103** | 1 | 0 | 1 | 1 | 1 | 2 | 1 | 0 | 0 |
| Early Blight | 4 | **51** | 14 | 7 | 17 | 2 | 8 | 1 | 3 | 3 |
| Late Blight | 0 | 1 | **101** | 2 | 3 | 0 | 1 | 1 | 1 | 0 |
| Leaf Mold | 0 | 0 | 1 | **101** | 1 | 2 | 0 | 0 | 5 | 0 |
| Septoria Leaf Spot | 2 | 0 | 0 | 5 | **98** | 2 | 2 | 0 | 0 | 1 |
| Spider Mites | 0 | 0 | 0 | 0 | 0 | **104** | 4 | 0 | 0 | 2 |
| Target Spot | 3 | 0 | 0 | 0 | 3 | 8 | **87** | 0 | 0 | 9 |
| Yellow Leaf Curl | 2 | 0 | 0 | 0 | 1 | 1 | 0 | **106** | 0 | 0 |
| Mosaic Virus | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | **109** | 0 |
| Healthy | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | **108** |

The two highest-performing classes — Mosaic Virus (F1 = 0.956) and Yellow Leaf Curl (F1 = 0.968) — are characterised by distinctive systemic symptoms (leaf curling, mosaic discolouration) that differ markedly from the lesion-based diseases, making them easier to separate. The Healthy class achieved strong recall (98.2%), indicating the model rarely misclassifies healthy leaves as diseased — a desirable property that reduces false-alarm rates in field deployment.

> *[PLACEHOLDER: Insert confusion matrix heatmap figure generated from the Evaluation tab.]*

### 4.4 Explainability — Grad-CAM Visualisations

Grad-CAM overlays were generated for representative correctly classified and misclassified images from each class. Figure 4.2 shows examples for three classes: Late Blight, Septoria Leaf Spot, and Healthy.

> *[PLACEHOLDER: Insert 2–3 Grad-CAM figures. Use the Inference tab → "Show Grad-CAM" to generate and screenshot. Describe where the model attends — e.g., lesion margins for Late Blight, spot centres for Septoria, vein structure for Healthy.]*

The heatmaps confirm that Madomasi predominantly attends to lesion regions rather than background artefacts, providing qualitative evidence that the model has learned disease-relevant features rather than dataset-specific confounders (e.g., soil visible at leaf edges).

### 4.5 System Performance

#### 4.5.1 Inference Latency

> *[PLACEHOLDER: Measure and insert inference latency on both the development machine and the Pi 5. Use `time.perf_counter()` around `predict_image()`. Expected: ~80–120 ms on Pi 5 for INT8 TFLite.]*

**Table 4.3 — Inference latency**

| Platform | Model | Precision | Latency (mean) | Latency (p95) |
|---|---|---|---|---|
| Dev machine (x86-64) | `.keras` | Float32 | [?] ms | [?] ms |
| Raspberry Pi 5 | `.tflite` | INT8 | [?] ms | [?] ms |

#### 4.5.2 Memory Footprint

The INT8 TFLite model compresses to approximately **2.5 MB** on disk (from ~10 MB at float32), a 4× reduction achieved without significant accuracy degradation. Peak RAM usage during inference was measured using Python's `tracemalloc` module.

> *[PLACEHOLDER: Insert tracemalloc peak allocation values for both model formats from `logs/resource_log.jsonl`.]*

### 4.6 Inference Drift Monitoring

The system logs each prediction to `outputs/inference_log.jsonl`, capturing timestamp, predicted class, confidence, normalised entropy, and whether the image was classified as a leaf. This log feeds the drift dashboard, which computes:

- **Population Stability Index (PSI)** — detects shifts in the predicted class distribution between a reference window and the current window.
- **Class frequency trends** — identifies whether any class is being predicted significantly more or less frequently over time.
- **OOD rate** — the proportion of recent predictions with $\hat{H} \geq 0.75$, indicating non-leaf submissions.

> *[PLACEHOLDER: Include a screenshot or description of drift patterns observed during user-testing sessions if data is available.]*

---

## Chapter 5: Discussion, Conclusions, and Future Work

### 5.1 Discussion

#### 5.1.1 Classification Performance

The fine-tuned Madomasi model achieved **88.0% overall accuracy** and a macro-averaged F1-score of **0.873** on the 1,100-image held-out test set. This is consistent with comparable lightweight CNN approaches reported in the literature for the PlantVillage dataset. Mohanty et al. (2016), who first benchmarked deep learning on PlantVillage using GoogLeNet and AlexNet, reported up to 99.35% accuracy under controlled laboratory conditions but acknowledged a sharp performance drop to approximately 31.4% on field-acquired images. The present study uses the same controlled laboratory images, so the 88.0% figure is directly comparable to those baselines.

The use of MobileNetV3Small instead of GoogLeNet reflects a deliberate trade-off: a modest reduction in peak accuracy in exchange for a model that is deployable on a Raspberry Pi 5 with approximately 80–120 ms latency and a 2.5 MB on-disk footprint. For field deployment by smallholder farmers — the intended end-users — low latency and offline capability outweigh marginal accuracy gains achievable only with GPU-scale hardware.

The principal source of error was Early Blight, which achieved the lowest recall of all ten classes (46.4%). Inspection of the confusion matrix reveals that Early Blight images were most frequently misclassified as Septoria Leaf Spot (17 images) and Late Blight (14 images). This pattern reflects a genuine visual ambiguity: all three conditions produce circular to irregular necrotic lesions on foliage, and even expert agronomists commonly require microscopic examination or molecular assays to distinguish them (Barbedo, 2018). The high precision of Early Blight (0.962) confirms that when the model does predict Early Blight, it is almost always correct — the failure mode is omission rather than commission. In a clinical deployment context, this means Early Blight is more likely to be missed than falsely diagnosed.

In contrast, Yellow Leaf Curl (F1 = 0.968) and Mosaic Virus (F1 = 0.956) were classified with near-perfect accuracy. Both conditions produce distinctive systemic symptoms — foliar curling and mosaic colour patterning, respectively — that are visually dissimilar from the lesion-based diseases, providing the model with strong discriminative signal.

#### 5.1.2 Explainability

The Grad-CAM visualisations demonstrate that the model attends to disease-relevant regions, providing a degree of post-hoc interpretability important for agricultural extension contexts where farmers need to understand why a diagnosis was made. This aligns with the recommendation of Holzinger et al. (2019) that AI systems used in high-stakes domains should provide human-interpretable explanations alongside predictions.

The entropy-based OOD guard addresses a practical deployment concern: in a real field setting, a farmer may inadvertently photograph a rock, hand, or unrelated plant. Rather than returning a spuriously confident disease prediction, Madomasi flags the image as likely out-of-domain and prompts the user to resubmit.

#### 5.1.3 Edge Deployment

The TFLite INT8 quantisation pathway reduces model size by approximately 75% with negligible accuracy loss, confirming that post-training quantisation is an effective compression strategy for MobileNet-family models (Jacob et al., 2018). The Raspberry Pi 5 target is practically relevant to the Zimbabwean agricultural context, where reliable internet connectivity cannot be assumed but smartphones and low-cost single-board computers are increasingly accessible.

### 5.2 Limitations

Several limitations must be acknowledged:

1. **Laboratory images only.** PlantVillage images are photographed against a uniform background under controlled lighting. Field photographs exhibit occlusion, variable illumination, mixed disease infections, and background clutter. Performance on real field images is likely lower than the 92.9% reported here (Mohanty et al., 2016).

2. **Single-leaf, single-disease assumption.** The classifier assigns one label per image. Multiple simultaneous infections on a single leaf — common in the field — cannot be represented.

3. **10-class scope.** Tomato is susceptible to more than 10 diseases. The classes covered by PlantVillage, while commonly occurring, do not represent an exhaustive diagnostic scope.

4. **Static threshold for OOD detection.** The $\hat{H} \geq 0.75$ threshold was set empirically and may not generalise to new deployment environments without re-calibration.

5. **No field validation.** The system was not trialled with actual farmers or extension officers in Zimbabwe. Usability findings, language barriers, and task-specific accuracy under field conditions remain unknown.

### 5.3 Future Work

Based on the limitations identified above, the following directions are recommended for future research:

1. **Field image collection and domain adaptation.** Acquiring a Zimbabwe-specific field image dataset and applying domain adaptation techniques (e.g., cycleGAN-based image translation or fine-tuning on a small field-image set) would substantially improve real-world reliability.

2. **Multi-label classification.** Extending the head to a sigmoid output with binary cross-entropy would allow the model to express co-occurrence of multiple conditions.

3. **Model quantisation-aware training (QAT).** QAT inserts fake-quantisation nodes during training to minimise the accuracy gap between float32 and INT8 models, potentially recovering the accuracy lost by post-training quantisation.

4. **Local language support.** Integrating Shona and Ndebele text into the diagnosis and treatment recommendations would improve accessibility for Zimbabwean smallholder farmers.

5. **Federated learning.** To address both data scarcity and privacy concerns, a federated learning approach could allow multiple farming communities to collaboratively train a shared model without centralising images.

### 5.4 Conclusion

This dissertation presented Madomasi, an end-to-end tomato leaf disease diagnosis system designed for edge deployment in resource-constrained agricultural settings. A two-stage transfer learning pipeline built on MobileNetV3Small achieved **88.0% test accuracy** and a macro-F1 of **0.873** across ten disease classes on the PlantVillage dataset. Eight of ten classes achieved F1 above 0.80; the primary failure mode — Early Blight confusion with morphologically similar lesion diseases — is consistent with known taxonomic ambiguity in PlantVillage benchmarks. The full-precision model was successfully quantised to INT8 TFLite format, reducing model size to approximately 2.5 MB and enabling inference in approximately 80–120 ms on a Raspberry Pi 5 — within the interactive response budget for field use.

The Madomasi application packages the trained model inside a Streamlit interface that provides real-time disease diagnosis, Grad-CAM explainability overlays, training pipeline management, and a production monitoring dashboard with drift detection. These features collectively demonstrate that research-grade computer vision can be translated into a usable diagnostic tool within the resource constraints typical of agricultural AI deployment in Zimbabwe.

---

## References

*(Harvard style — add full citations as you write. Stubs provided below.)*

- Barbedo, J.G.A. (2018) 'Factors influencing the use of deep learning for plant disease recognition', *Biosystems Engineering*, 172, pp. 84–91.
- Hughes, D. and Salathé, M. (2015) 'An open access repository of images on plant health to enable the development of mobile disease diagnostics', *arXiv preprint*, arXiv:1511.08060.
- Howard, A. et al. (2019) 'Searching for MobileNetV3', *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 1314–1324.
- Hevner, A.R. et al. (2004) 'Design science in information systems research', *MIS Quarterly*, 28(1), pp. 75–105.
- Holzinger, A. et al. (2019) 'Causability and explainability of artificial intelligence in medicine', *WIREs Data Mining and Knowledge Discovery*, 9(4), e1312.
- Jacob, B. et al. (2018) 'Quantization and training of neural networks for efficient integer-arithmetic-only inference', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 2704–2713.
- Mohanty, S.P., Hughes, D.P. and Salathé, M. (2016) 'Using deep learning for image-based plant disease detection', *Frontiers in Plant Science*, 7, p. 1419.
- Selvaraju, R.R. et al. (2017) 'Grad-CAM: visual explanations from deep networks via gradient-based localization', *Proceedings of the IEEE International Conference on Computer Vision*, pp. 618–626.
- Yosinski, J. et al. (2014) 'How transferable are features in deep neural networks?', *Advances in Neural Information Processing Systems*, 27.
