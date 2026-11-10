"""Reader / writer for the Live2D Cubism .moc3 runtime container (versions 1-3 layout).

The file is a flat set of arrays addressed through a section offset table.
Only the sections that exist up to moc version 3 (Cubism 4.0) are modelled; newer
files are still readable for those sections.
"""
from __future__ import annotations

import struct
from collections import OrderedDict

import numpy as np

HEADER_SIZE = 64
OFFSET_TABLE_ENTRIES = 160
COUNT_INFO_OFFSET = 0x7C0
ID_SIZE = 64
ALIGN = 64

COUNT_NAMES = [
    "parts", "deformers", "warp", "rot", "art", "params", "partKf", "warpKf", "rotKf",
    "artKf", "kfPos", "pbi", "kfb", "pb", "keys", "uvs", "posIdx", "masks", "dog", "dogo",
    "glue", "glueInfo", "glueKf",
]

# (group, field, kind) in section-offset-table order. kind: rt=8 bytes of runtime
# scratch per item, id=64 byte string, otherwise a numpy dtype string.
SECTIONS = [
    ("parts", "runtime0", "rt"), ("parts", "ids", "id"), ("parts", "kfbIdx", "<i4"),
    ("parts", "kfBegin", "<i4"), ("parts", "kfCount", "<i4"), ("parts", "isVisible", "<i4"),
    ("parts", "isEnabled", "<i4"), ("parts", "parentPart", "<i4"),

    ("deformers", "runtime0", "rt"), ("deformers", "ids", "id"), ("deformers", "kfbIdx", "<i4"),
    ("deformers", "isVisible", "<i4"), ("deformers", "isEnabled", "<i4"),
    ("deformers", "parentPart", "<i4"), ("deformers", "parentDeformer", "<i4"),
    ("deformers", "type", "<i4"), ("deformers", "specificIdx", "<i4"),

    ("warp", "kfbIdx", "<i4"), ("warp", "kfBegin", "<i4"), ("warp", "kfCount", "<i4"),
    ("warp", "vertexCount", "<i4"), ("warp", "rows", "<i4"), ("warp", "cols", "<i4"),

    ("rot", "kfbIdx", "<i4"), ("rot", "kfBegin", "<i4"), ("rot", "kfCount", "<i4"),
    ("rot", "baseAngle", "<f4"),

    ("art", "runtime0", "rt"), ("art", "runtime1", "rt"), ("art", "runtime2", "rt"),
    ("art", "runtime3", "rt"), ("art", "ids", "id"), ("art", "kfbIdx", "<i4"),
    ("art", "kfBegin", "<i4"), ("art", "kfCount", "<i4"), ("art", "isVisible", "<i4"),
    ("art", "isEnabled", "<i4"), ("art", "parentPart", "<i4"), ("art", "parentDeformer", "<i4"),
    ("art", "textureNo", "<i4"), ("art", "flags", "u1"), ("art", "vertexCount", "<i4"),
    ("art", "uvBegin", "<i4"), ("art", "idxBegin", "<i4"), ("art", "idxCount", "<i4"),
    ("art", "maskBegin", "<i4"), ("art", "maskCount", "<i4"),

    ("params", "runtime0", "rt"), ("params", "ids", "id"), ("params", "max", "<f4"),
    ("params", "min", "<f4"), ("params", "default", "<f4"), ("params", "isRepeat", "<i4"),
    ("params", "decimalPlaces", "<i4"), ("params", "pbBegin", "<i4"), ("params", "pbCount", "<i4"),

    ("partKf", "drawOrder", "<f4"),
    ("warpKf", "opacity", "<f4"), ("warpKf", "posBegin", "<i4"),
    ("rotKf", "opacity", "<f4"), ("rotKf", "angle", "<f4"), ("rotKf", "originX", "<f4"),
    ("rotKf", "originY", "<f4"), ("rotKf", "scale", "<f4"), ("rotKf", "reflectX", "<i4"),
    ("rotKf", "reflectY", "<i4"),
    ("artKf", "opacity", "<f4"), ("artKf", "drawOrder", "<f4"), ("artKf", "posBegin", "<i4"),
    ("kfPos", "xy", "<f4"),
    ("pbi", "idx", "<i4"),
    ("kfb", "pbiBegin", "<i4"), ("kfb", "pbiCount", "<i4"),
    ("pb", "keysBegin", "<i4"), ("pb", "keysCount", "<i4"),
    ("keys", "value", "<f4"),
    ("uvs", "uv", "<f4"),
    ("posIdx", "idx", "<i2"),
    ("masks", "drawableIdx", "<i4"),
    ("dog", "objBegin", "<i4"), ("dog", "objCount", "<i4"), ("dog", "objTotalCount", "<i4"),
    ("dog", "maxDrawOrder", "<i4"), ("dog", "minDrawOrder", "<i4"),
    ("dogo", "type", "<i4"), ("dogo", "index", "<i4"), ("dogo", "selfIndex", "<i4"),
    ("glue", "runtime0", "rt"), ("glue", "ids", "id"), ("glue", "kfbIdx", "<i4"),
    ("glue", "kfBegin", "<i4"), ("glue", "kfCount", "<i4"), ("glue", "artA", "<i4"),
    ("glue", "artB", "<i4"), ("glue", "infoBegin", "<i4"), ("glue", "infoCount", "<i4"),
    ("glueInfo", "weight", "<f4"), ("glueInfo", "posIdx", "<i2"),
    ("glueKf", "intensity", "<f4"),
    # moc version >= 2 (Cubism 3.3)
    ("warp", "isQuadSource", "<i4"),
]
SECTIONS_V1 = len(SECTIONS) - 1


def _item_size(kind: str) -> int:
    if kind == "rt":
        return 8
    if kind == "id":
        return ID_SIZE
    return np.dtype(kind).itemsize


def _read_ids(buf: bytes, off: int, n: int) -> list[str]:
    out = []
    for i in range(n):
        raw = buf[off + i * ID_SIZE: off + (i + 1) * ID_SIZE]
        out.append(raw.split(b"\0", 1)[0].decode("utf-8"))
    return out


def parse(path: str) -> dict:
    buf = open(path, "rb").read()
    assert buf[:4] == b"MOC3", "not a moc3 file"
    version = buf[4]
    offs = struct.unpack_from(f"<{OFFSET_TABLE_ENTRIES}I", buf, HEADER_SIZE)
    counts_raw = struct.unpack_from("<32I", buf, offs[0])
    counts = OrderedDict(zip(COUNT_NAMES, counts_raw))
    ppu, ox, oy, cw, ch = struct.unpack_from("<5f", buf, offs[1])
    model: dict = {
        "version": version,
        "counts": counts,
        "canvas": {"ppu": ppu, "originX": ox, "originY": oy, "width": cw, "height": ch,
                   "flags": buf[offs[1] + 20]},
        "offsets": offs,
        "size": len(buf),
    }
    nsec = SECTIONS_V1 if version < 2 else len(SECTIONS)
    for i, (group, field, kind) in enumerate(SECTIONS[:nsec]):
        off = offs[2 + i]
        n = counts[group]
        g = model.setdefault(group, {})
        if kind == "rt":
            continue
        if kind == "id":
            g[field] = _read_ids(buf, off, n)
        else:
            g[field] = np.frombuffer(buf, dtype=kind, count=n, offset=off).copy()
    return model


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def write(model: dict, path: str, version: int = 3) -> None:
    """Serialise `model` (same structure as parse()) to a moc3 file."""
    counts = OrderedDict()
    for name in COUNT_NAMES:
        counts[name] = 0
    nsec = SECTIONS_V1 if version < 2 else len(SECTIONS)
    arrays = []
    for group, field, kind in SECTIONS[:nsec]:
        g = model.get(group, {})
        if kind == "rt":
            arrays.append(None)
            continue
        data = g.get(field)
        if data is None:
            data = [] if kind == "id" else np.zeros(0, dtype=kind)
        if kind == "id":
            n = len(data)
        else:
            data = np.ascontiguousarray(data, dtype=kind)
            n = data.shape[0]
        if counts[group] == 0:
            counts[group] = n
        assert counts[group] == n, f"count mismatch in {group}.{field}: {n} vs {counts[group]}"
        arrays.append(data)

    out = bytearray(COUNT_INFO_OFFSET)
    out[0:4] = b"MOC3"
    out[4] = version
    out[5] = 0
    offs = [0] * OFFSET_TABLE_ENTRIES

    # count info (128 bytes) + canvas info (64 bytes)
    offs[0] = len(out)
    ci = bytearray(128)
    struct.pack_into(f"<{len(COUNT_NAMES)}I", ci, 0, *[counts[n] for n in COUNT_NAMES])
    out += ci
    offs[1] = len(out)
    cv = model["canvas"]
    cb = bytearray(64)
    struct.pack_into("<5f", cb, 0, cv["ppu"], cv["originX"], cv["originY"], cv["width"], cv["height"])
    cb[20] = cv.get("flags", 0)
    out += cb

    prev_kind = None
    for i, (group, field, kind) in enumerate(SECTIONS[:nsec]):
        n = counts[group]
        # id tables follow their runtime block without padding; everything else is 64-aligned
        if not (kind == "id" and prev_kind == "rt"):
            out += bytes(_align(len(out)) - len(out))
        offs[2 + i] = len(out)
        if kind == "rt":
            out += bytes(8 * n)
        elif kind == "id":
            for s in arrays[i]:
                raw = s.encode("utf-8")
                assert len(raw) < ID_SIZE, f"id too long: {s}"
                out += raw + bytes(ID_SIZE - len(raw))
        else:
            out += arrays[i].tobytes()
        prev_kind = kind
    out += bytes(_align(len(out)) - len(out))
    struct.pack_into(f"<{OFFSET_TABLE_ENTRIES}I", out, HEADER_SIZE, *offs)
    with open(path, "wb") as fh:
        fh.write(out)
