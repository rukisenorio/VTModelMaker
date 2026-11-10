"""ctypes binding to the Live2D Cubism Core - the one VTube Studio ships (or one set in ~/.vtmm/config.json "core_lib").

Used only to evaluate/validate generated models with the same runtime VTube Studio uses. The Core is never copied or
shipped with this tool: it is loaded from where VTube Studio installed it.
"""
from __future__ import annotations

import ctypes as C
import os
import sys

import numpy as np

try:
    import platform_env as _pe
except ImportError:                                        # (imported from outside the engine folder)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import platform_env as _pe

CORE_PATH = None
_lib = None


def lib():
    global _lib
    global CORE_PATH
    if _lib is None:
        CORE_PATH = _pe.core_lib()
        if not CORE_PATH:
            raise RuntimeError("Live2D Cubism Core が見つかりません。VTube Studio（Steam）をインストールするか、"
                               + _pe.CONFIG_PATH + " の \"core_lib\" に場所を書いてください")
        L = C.CDLL(CORE_PATH)
        L.csmGetVersion.restype = C.c_uint
        L.csmGetLatestMocVersion.restype = C.c_uint
        L.csmGetMocVersion.restype = C.c_uint
        L.csmGetMocVersion.argtypes = [C.c_void_p, C.c_uint]
        L.csmHasMocConsistency.restype = C.c_int
        L.csmHasMocConsistency.argtypes = [C.c_void_p, C.c_uint]
        L.csmReviveMocInPlace.restype = C.c_void_p
        L.csmReviveMocInPlace.argtypes = [C.c_void_p, C.c_uint]
        L.csmGetSizeofModel.restype = C.c_uint
        L.csmGetSizeofModel.argtypes = [C.c_void_p]
        L.csmInitializeModelInPlace.restype = C.c_void_p
        L.csmInitializeModelInPlace.argtypes = [C.c_void_p, C.c_void_p, C.c_uint]
        L.csmUpdateModel.argtypes = [C.c_void_p]
        L.csmResetDrawableDynamicFlags.argtypes = [C.c_void_p]
        L.csmReadCanvasInfo.argtypes = [C.c_void_p, C.POINTER(C.c_float * 2),
                                        C.POINTER(C.c_float * 2), C.POINTER(C.c_float)]
        for name in ("csmGetParameterCount", "csmGetPartCount", "csmGetDrawableCount"):
            getattr(L, name).restype = C.c_int
            getattr(L, name).argtypes = [C.c_void_p]
        for name in (
            "csmGetParameterIds", "csmGetParameterMinimumValues", "csmGetParameterMaximumValues",
            "csmGetParameterDefaultValues", "csmGetParameterValues", "csmGetParameterKeyCounts",
            "csmGetParameterKeyValues", "csmGetPartIds", "csmGetPartOpacities",
            "csmGetPartParentPartIndices", "csmGetDrawableIds", "csmGetDrawableConstantFlags",
            "csmGetDrawableDynamicFlags", "csmGetDrawableTextureIndices", "csmGetDrawableDrawOrders",
            "csmGetDrawableRenderOrders", "csmGetDrawableOpacities", "csmGetDrawableMaskCounts",
            "csmGetDrawableMasks", "csmGetDrawableVertexCounts", "csmGetDrawableVertexPositions",
            "csmGetDrawableVertexUvs", "csmGetDrawableIndexCounts", "csmGetDrawableIndices",
            "csmGetDrawableParentPartIndices",
        ):
            getattr(L, name).restype = C.c_void_p
            getattr(L, name).argtypes = [C.c_void_p]
        _lib = L
    return _lib


def _aligned(size: int, align: int):
    raw = (C.c_char * (size + align))()
    addr = C.addressof(raw)
    pad = (-addr) % align
    return raw, addr + pad


def _arr(ptr, ctype, n):
    if not ptr or n == 0:
        return np.zeros(0, dtype=np.dtype(ctype))
    return np.ctypeslib.as_array(C.cast(ptr, C.POINTER(ctype)), shape=(n,))


def _strs(ptr, n):
    p = C.cast(ptr, C.POINTER(C.c_char_p))
    return [p[i].decode("utf-8") for i in range(n)]


def core_version() -> str:
    v = lib().csmGetVersion()
    return f"{v >> 24}.{(v >> 16) & 0xFF}.{v & 0xFFFF}"


def check(path: str) -> dict:
    """Version + consistency check without instantiating a model."""
    L = lib()
    data = open(path, "rb").read()
    raw, addr = _aligned(len(data), 64)
    C.memmove(addr, data, len(data))
    return {
        "core": core_version(),
        "latest_moc_version": L.csmGetLatestMocVersion(),
        "moc_version": L.csmGetMocVersion(addr, len(data)),
        "consistent": bool(L.csmHasMocConsistency(addr, len(data))),
    }


class Model:
    def __init__(self, path: str, check_consistency: bool = True):
        L = lib()
        data = open(path, "rb").read()
        self._moc_raw, moc_addr = _aligned(len(data), 64)
        C.memmove(moc_addr, data, len(data))
        if check_consistency and not L.csmHasMocConsistency(moc_addr, len(data)):
            raise ValueError(f"moc3 が Core の整合性チェックに通りませんでした: {path}")
        self.moc = L.csmReviveMocInPlace(moc_addr, len(data))
        if not self.moc:
            raise ValueError("csmReviveMocInPlace に失敗しました")
        size = L.csmGetSizeofModel(self.moc)
        self._model_raw, model_addr = _aligned(size, 16)
        self.model = L.csmInitializeModelInPlace(self.moc, model_addr, size)
        if not self.model:
            raise ValueError("csmInitializeModelInPlace に失敗しました")
        m = self.model
        size_px = (C.c_float * 2)()
        origin_px = (C.c_float * 2)()
        ppu = C.c_float()
        L.csmReadCanvasInfo(m, C.byref(size_px), C.byref(origin_px), C.byref(ppu))
        self.canvas_size = (size_px[0], size_px[1])
        self.canvas_origin = (origin_px[0], origin_px[1])
        self.ppu = ppu.value

        np_ = L.csmGetParameterCount(m)
        self.param_ids = _strs(L.csmGetParameterIds(m), np_)
        self.param_min = _arr(L.csmGetParameterMinimumValues(m), C.c_float, np_).copy()
        self.param_max = _arr(L.csmGetParameterMaximumValues(m), C.c_float, np_).copy()
        self.param_default = _arr(L.csmGetParameterDefaultValues(m), C.c_float, np_).copy()
        self.param_values = _arr(L.csmGetParameterValues(m), C.c_float, np_)  # live view
        self.param_index = {k: i for i, k in enumerate(self.param_ids)}

        npart = L.csmGetPartCount(m)
        self.part_ids = _strs(L.csmGetPartIds(m), npart)
        self.part_opacities = _arr(L.csmGetPartOpacities(m), C.c_float, npart)
        self.part_parents = _arr(L.csmGetPartParentPartIndices(m), C.c_int, npart).copy()

        nd = L.csmGetDrawableCount(m)
        self.n_drawables = nd
        self.drawable_ids = _strs(L.csmGetDrawableIds(m), nd)
        self.constant_flags = _arr(L.csmGetDrawableConstantFlags(m), C.c_ubyte, nd).copy()
        self._dynamic_flags = _arr(L.csmGetDrawableDynamicFlags(m), C.c_ubyte, nd)
        self.texture_indices = _arr(L.csmGetDrawableTextureIndices(m), C.c_int, nd).copy()
        self._draw_orders = _arr(L.csmGetDrawableDrawOrders(m), C.c_int, nd)
        self._render_orders = _arr(L.csmGetDrawableRenderOrders(m), C.c_int, nd)
        self._opacities = _arr(L.csmGetDrawableOpacities(m), C.c_float, nd)
        self.vertex_counts = _arr(L.csmGetDrawableVertexCounts(m), C.c_int, nd).copy()
        self.index_counts = _arr(L.csmGetDrawableIndexCounts(m), C.c_int, nd).copy()
        self.mask_counts = _arr(L.csmGetDrawableMaskCounts(m), C.c_int, nd).copy()
        self.drawable_parts = _arr(L.csmGetDrawableParentPartIndices(m), C.c_int, nd).copy()
        pos_pp = C.cast(L.csmGetDrawableVertexPositions(m), C.POINTER(C.c_void_p))
        uv_pp = C.cast(L.csmGetDrawableVertexUvs(m), C.POINTER(C.c_void_p))
        idx_pp = C.cast(L.csmGetDrawableIndices(m), C.POINTER(C.c_void_p))
        mask_pp = C.cast(L.csmGetDrawableMasks(m), C.POINTER(C.c_void_p))
        self._pos, self.uvs, self.indices, self.masks = [], [], [], []
        for i in range(nd):
            vc = int(self.vertex_counts[i])
            self._pos.append(_arr(pos_pp[i], C.c_float, vc * 2).reshape(-1, 2))
            self.uvs.append(_arr(uv_pp[i], C.c_float, vc * 2).reshape(-1, 2).copy())
            self.indices.append(_arr(idx_pp[i], C.c_ushort, int(self.index_counts[i])).copy())
            self.masks.append(_arr(mask_pp[i], C.c_int, int(self.mask_counts[i])).copy())
        self.update()

    def set(self, **values):
        for k, v in values.items():
            self.param_values[self.param_index[k]] = v

    def reset(self):
        self.param_values[:] = self.param_default

    def update(self):
        L = lib()
        L.csmResetDrawableDynamicFlags(self.model)
        L.csmUpdateModel(self.model)

    # live views -----------------------------------------------------------------
    def positions(self, i: int) -> np.ndarray:
        """Vertex positions of drawable i in Core units (x right, y up, origin = canvas origin)."""
        return self._pos[i]

    def positions_px(self, i: int) -> np.ndarray:
        p = self._pos[i]
        out = np.empty_like(p)
        out[:, 0] = p[:, 0] * self.ppu + self.canvas_origin[0]
        out[:, 1] = -p[:, 1] * self.ppu + self.canvas_origin[1]
        return out

    @property
    def opacities(self):
        return self._opacities

    @property
    def render_orders(self):
        return self._render_orders

    @property
    def draw_orders(self):
        return self._draw_orders

    @property
    def dynamic_flags(self):
        return self._dynamic_flags
