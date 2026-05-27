import numpy as np


def grabcut_bg_remove(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace background pixels with white (255, 255, 255) using GrabCut.

    White matches the PlantVillage training-image background, preventing the
    domain shift that neutral grey would introduce at inference time.

    Assumes the subject is centred — guaranteed by the browser square-crop step.

    Returns:
        segmented_rgb  — uint8 RGB array, same shape as input
        fg_mask        — uint8 binary mask (1 = foreground, 0 = background)
    """
    import cv2

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pad = max(8, min(h, w) // 10)
    rect = (pad, pad, w - 2 * pad, h - 2 * pad)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where(
            (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1
        ).astype(np.uint8)
        result = img_rgb.copy()
        result[fg_mask == 0] = [255, 255, 255]
        return result, fg_mask
    except Exception:
        return img_rgb, np.ones((h, w), np.uint8)
