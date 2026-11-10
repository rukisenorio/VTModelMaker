"""SAM-HQ wrapper: prompt-driven masks on crops of the full-resolution illustration.

All coordinates given to / returned from this module are canvas pixels.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # the weights are already in the local cache

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MODEL_ID = "syscv-community/sam-hq-vit-huge"


def _patch_decoder():
    """transformers 4.57.x drops the attended image embeddings in SamHQMaskDecoder.forward and
    then transposes H/W of the stale tensor. Rebind the transformer output correctly."""
    import inspect
    import textwrap
    import transformers.models.sam_hq.modeling_sam_hq as mod
    cls = mod.SamHQMaskDecoder
    if getattr(cls, "_l2d_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(cls.forward))
    bad = "point_embedding, iou_token_out = self.transformer("
    if bad in src:
        src = src.replace(bad, "point_embedding, image_embeddings = self.transformer(")
        ns = {}
        exec(compile(src, mod.__file__, "exec"), mod.__dict__, ns)
        cls.forward = ns["forward"]
    cls._l2d_patched = True


class Sam:
    def __init__(self, image: np.ndarray | None = None, device: str | None = None):
        from transformers import SamHQModel, SamHQProcessor
        _patch_decoder()
        import platform_env
        self.device = device or platform_env.torch_device()
        self.model = SamHQModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self.proc = SamHQProcessor.from_pretrained(MODEL_ID)
        self.image = image
        self._emb = {}

    def _embed(self, image, box):
        key = (id(image), tuple(box))
        if key not in self._emb:
            x0, y0, x1, y1 = box
            crop = image[y0:y1, x0:x1]
            inputs = self.proc(images=Image.fromarray(crop), return_tensors="pt")
            pv = inputs["pixel_values"].to(self.device, torch.float32)
            with torch.no_grad():
                emb = self.model.get_image_embeddings(pv)
            self._emb[key] = (emb, inputs["original_sizes"], inputs["reshaped_input_sizes"])
            if len(self._emb) > 8:
                self._emb.pop(next(iter(self._emb)))
        return self._emb[key]

    def predict(self, crop_box, pos=(), neg=(), box=None, multimask=True, pick="score", image=None, origin=(0, 0)):
        """crop_box: (x0,y0,x1,y1) window fed to SAM, in canvas px. pos/neg: lists of (x,y) canvas px.
        image: optional alternative image whose pixel (0,0) sits at canvas `origin`.

        Returns (mask bool array of crop size, score, all_masks, all_scores).
        """
        img = self.image if image is None else image
        ox, oy = origin
        x0, y0, x1, y1 = crop_box
        lbox = (x0 - ox, y0 - oy, x1 - ox, y1 - oy)
        emb, orig, resh = self._embed(img, lbox)
        image_emb, inter_emb = emb
        pts = [(px - x0, py - y0) for px, py in pos] + [(px - x0, py - y0) for px, py in neg]
        labels = [1] * len(pos) + [0] * len(neg)
        kwargs = {}
        crop = img[lbox[1]:lbox[3], lbox[0]:lbox[2]]
        enc = self.proc(
            images=Image.fromarray(crop),
            input_points=[[pts]] if pts else None,
            input_labels=[[labels]] if pts else None,
            input_boxes=[[[box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]]] if box else None,
            return_tensors="pt",
        )
        for k in ("input_points", "input_labels", "input_boxes"):
            if k in enc and enc[k] is not None:
                t = enc[k]
                t = t.to(torch.float32) if t.is_floating_point() else t
                kwargs[k] = t.to(self.device)
        with torch.no_grad():
            out = self.model(image_embeddings=image_emb, intermediate_embeddings=inter_emb,
                             multimask_output=multimask, **kwargs)
        masks = self.proc.image_processor.post_process_masks(out.pred_masks.cpu(), orig, resh)[0]
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1]).numpy().astype(bool)
        scores = out.iou_scores.cpu().numpy().reshape(-1)
        if pick == "score":
            i = int(np.argmax(scores))
        elif pick == "largest":
            i = int(np.argmax(masks.reshape(len(masks), -1).sum(1)))
        elif pick == "smallest":
            i = int(np.argmin(masks.reshape(len(masks), -1).sum(1)))
        else:
            i = int(pick)
        return masks[i], float(scores[i]), masks, scores

    def mask(self, window, crop_box, **kw):
        """Run predict on crop_box and return the mask placed in `window` (canvas box) coordinates."""
        m, score, _, _ = self.predict(crop_box, **kw)
        wx0, wy0, wx1, wy1 = window
        out = np.zeros((wy1 - wy0, wx1 - wx0), bool)
        x0, y0, x1, y1 = crop_box
        ix0, iy0, ix1, iy1 = max(x0, wx0), max(y0, wy0), min(x1, wx1), min(y1, wy1)
        out[iy0 - wy0:iy1 - wy0, ix0 - wx0:ix1 - wx0] = m[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        return out, score
