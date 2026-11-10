"""Register a generated edit back onto the source crop it was made from."""
import cv2
import numpy as np


def register(gen_rgb: np.ndarray, src_rgb: np.ndarray, valid_src: np.ndarray | None = None,
             model: str = "affine", flow_refine: bool = True):
    """Warp `gen_rgb` into the pixel grid of `src_rgb`.

    valid_src: bool mask (src size) of pixels that are expected to be unchanged by the edit;
               used both for feature matching and for the dense refinement.
    Returns (warped_gen uint8, info dict).
    """
    h, w = src_rgb.shape[:2]
    gen_up = cv2.resize(gen_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    g1 = cv2.cvtColor(gen_up, cv2.COLOR_RGB2GRAY)
    g0 = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    m0 = None if valid_src is None else (valid_src.astype(np.uint8) * 255)
    k0, d0 = sift.detectAndCompute(g0, m0)
    k1, d1 = sift.detectAndCompute(g1, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for pair in matcher.knnMatch(d1, d0, k=2):
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
            good.append(pair[0])
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p0 = np.float32([k0[m.trainIdx].pt for m in good])
    # keep only matches that are roughly in place (the edit preserves framing)
    near = np.linalg.norm(p1 - p0, axis=1) < 0.06 * max(w, h)
    p1, p0 = p1[near], p0[near]
    if model == "homography":
        H, inl = cv2.findHomography(p1, p0, cv2.RANSAC, 2.0)
        warped = cv2.warpPerspective(gen_up, H, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        A = H
    else:
        A, inl = cv2.estimateAffine2D(p1, p0, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        warped = cv2.warpAffine(gen_up, A, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    info = {"matches": int(len(p1)), "inliers": int(inl.sum()), "A": np.asarray(A).round(5).tolist()}
    if flow_refine:
        # dense residual flow estimated on the unchanged area, smoothly extended elsewhere
        wg = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(1)
        flow = dis.calc(g0, wg, None)   # for each src pixel, where it is in warped gen
        if valid_src is not None:
            diff = np.abs(g0.astype(np.int16) - cv2.remap(wg, *_maps(flow), cv2.INTER_LINEAR).astype(np.int16))
            trust = (valid_src & (diff < 28)).astype(np.float32)
        else:
            trust = np.ones((h, w), np.float32)
        mag = np.linalg.norm(flow, axis=2)
        trust *= (mag < 12).astype(np.float32)
        sigma = max(w, h) * 0.02
        num = cv2.GaussianBlur(flow * trust[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(trust, (0, 0), sigma)[..., None]
        smooth = num / np.maximum(den, 1e-4)
        # second, wider pass fills regions with no trusted samples
        num2 = cv2.GaussianBlur(flow * trust[..., None], (0, 0), sigma * 4)
        den2 = cv2.GaussianBlur(trust, (0, 0), sigma * 4)[..., None]
        wide = num2 / np.maximum(den2, 1e-4)
        conf = np.clip(den * 4, 0, 1)
        smooth = smooth * conf + wide * (1 - conf)
        warped = cv2.remap(warped, *_maps(smooth), cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        info["flow_px_mean"] = float(np.linalg.norm(smooth, axis=2).mean())
        info["flow_px_max"] = float(np.linalg.norm(smooth, axis=2).max())
    return warped, info


def _maps(flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return gx + flow[..., 0], gy + flow[..., 1]
