"""A 2x upscaler learned from this illustration itself.

There is no larger original of the artwork, and plain interpolation adds nothing. But the drawing
holds the same kinds of lines and edges at every size, so a small network can learn "what a line
of this artist looks like at twice the resolution" from the artwork alone: the artwork reduced to
half size is the input, the artwork itself the answer (zero-shot super-resolution). Applied to
the artwork at its own size it gives twice the size, with edges as crisp as the artwork's own.

usage: sr2x.py <project> train [iterations]   (or with L2D_PROJECT set: sr2x.py train)
  train    learn from the project's src/art.png; writes work/sr/net.pt and prints how well three held-back
           patches are recovered from their half-size copies, against plain cubic interpolation
           (minutes on a GPU or Apple silicon; slow on a plain CPU)
  up(img)  2x of an RGB(A) image (function for the other tools); without a net, plain Lanczos
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.environ.get("L2D_PROJECT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "work", "sr", "net.pt")
if not os.path.exists(NET):      # (the kit: the net trained on the first character, shipped with the skill)
    for cand in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sr_net.pt"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr_net.pt")):
        if os.path.exists(cand):
            NET = cand
            break
import platform_env
DEV = platform_env.torch_device()
HOLD = [(2128, 849, 2640, 1361), (2500, 6850, 2950, 7200), (420, 3280, 800, 3560)]      # held back from training: right eye, a boot, a hand


class Net(nn.Module):
    """Cubic 2x plus a learned correction (8 conv layers, 64 channels)."""
    def __init__(self, ch=64, depth=8):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 3, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, lr):
        base = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        return base + self.body(base)


def half(img):
    """Area reduction to half size (float32 0..1, HxWx3)."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2, -1).mean((1, 3))


def psnr(a, b):
    return float(-10 * np.log10(max(float(((a - b) ** 2).mean()), 1e-12)))


def apply_net(net, lr, tile=512, pad=16):
    """lr: float32 HxWx3 0..1 -> 2x, tile by tile."""
    h, w = lr.shape[:2]
    out = np.zeros((h * 2, w * 2, 3), np.float32)
    with torch.no_grad():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y0, x0, y1, x1 = max(y - pad, 0), max(x - pad, 0), min(y + tile + pad, h), min(x + tile + pad, w)
                t = torch.from_numpy(np.ascontiguousarray(lr[y0:y1, x0:x1].transpose(2, 0, 1)))[None].to(DEV)
                o = net(t)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                ya, xa = (y - y0) * 2, (x - x0) * 2
                yb, xb = ya + (min(y + tile, h) - y) * 2, xa + (min(x + tile, w) - x) * 2
                out[y * 2:y * 2 + (yb - ya), x * 2:x * 2 + (xb - xa)] = o[ya:yb, xa:xb]
    return out


_net = None


def load():
    """The 2x net, or None when there is none (no shipped net, none trained for this project)."""
    global _net
    if _net is None:
        if not os.path.exists(NET):
            print("sr2x: 2倍に拡大する学習済みデータ（work/sr/net.pt）がないので、ふつうの拡大（Lanczos）にします。"
                  "`sr2x.py <project> train` で絵から学習できます", flush=True)
            _net = False
            return None
        _net = Net().to(DEV)
        _net.load_state_dict(torch.load(NET, map_location=DEV))
        _net.eval()
    return _net or None


def _lanczos2x(img):
    return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)


def up(img):
    """2x of a uint8 image: RGB, or RGBA (colours first carried into the transparent parts, so that
    no dark fringe is learned into the edge; coverage upscaled as a picture of its own)."""
    net = load()
    if net is None:
        out = _lanczos2x(img)
        if img.shape[2] == 4:
            out[out[..., 3] == 0] = 0
        return out
    if img.shape[2] == 3:
        return np.clip(apply_net(net, img.astype(np.float32) / 255.0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    from scipy import ndimage
    a = img[..., 3]
    rgb = img[..., :3]
    known = a > 0
    if not known.any():
        return np.zeros((img.shape[0] * 2, img.shape[1] * 2, 4), np.uint8)
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    rgb = rgb[idx[0], idx[1]]
    rgb2 = apply_net(net, rgb.astype(np.float32) / 255.0)
    a3 = np.repeat(a[..., None], 3, 2).astype(np.float32) / 255.0
    a2 = apply_net(net, a3).mean(2)
    out = np.dstack([rgb2, np.clip(a2, 0, 1)])
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    return out


def train(iters=6000, bs=24, ps=64):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    A = np.asarray(Image.open(os.path.join(ROOT, "src", "art.png")).convert("RGB")).astype(np.float32) / 255.0
    extra = []
    global HOLD
    _H, _W = A.shape[:2]
    _r = np.random.default_rng(1)
    _s = max(64, min(_H, _W) // 10)
    HOLD = [(int(x), int(y), int(x) + _s, int(y) + _s) for x, y in zip(_r.integers(0, _W - _s, 3), _r.integers(0, _H - _s, 3))]
    H, W = A.shape[:2]
    holdm = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in HOLD:
        holdm[y0:y1, x0:x1] = True
    # where there is something to learn: not the empty page
    g = cv2.cvtColor((A * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    busy = cv2.dilate((cv2.Laplacian(g, cv2.CV_32F, ksize=3).__abs__() > 12).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ok = busy & ~holdm
    ys, xs = np.nonzero(ok[ps:H - ps, ps:W - ps])
    ys, xs = ys + ps, xs + ps
    rng = np.random.default_rng(0)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.6), int(iters * 0.85)], 0.3)
    t0 = time.time()
    HP = ps * 2
    for it in range(1, iters + 1):
        hr = np.zeros((bs, HP, HP, 3), np.float32)
        for b in range(bs):
            if extra and rng.random() < 0.04:
                E = extra[int(rng.integers(len(extra)))]
                y = int(rng.integers(0, E.shape[0] - HP)); x = int(rng.integers(0, E.shape[1] - HP))
                p = E[y:y + HP, x:x + HP]
            else:
                k = int(rng.integers(len(ys)))
                y, x = int(ys[k]) - ps, int(xs[k]) - ps
                p = A[y:y + HP, x:x + HP]
            r = int(rng.integers(8))
            p = np.rot90(p, r % 4)
            if r >= 4:
                p = p[:, ::-1]
            hr[b] = p
        lr_ = hr.reshape(bs, ps, 2, ps, 2, 3).mean((2, 4))
        t_hr = torch.from_numpy(hr.transpose(0, 3, 1, 2).copy()).to(DEV)
        t_lr = torch.from_numpy(lr_.transpose(0, 3, 1, 2).copy()).to(DEV)
        loss = F.l1_loss(net(t_lr), t_hr)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0:
            print(f"学習 {it} 回目: 誤差 {loss.item():.5f}（{time.time() - t0:.0f} 秒）", flush=True)
            os.makedirs(os.path.dirname(NET), exist_ok=True)
            torch.save(net.state_dict(), NET + ".part")          # (kept while training, in case the run is cut short)
    os.makedirs(os.path.dirname(NET), exist_ok=True)
    torch.save(net.state_dict(), NET)
    net.eval()
    for (x0, y0, x1, y1), nm in zip(HOLD, ("held-back 1", "held-back 2", "held-back 3")):
        hrp = A[y0:y1, x0:x1]
        hrp = hrp[:hrp.shape[0] // 2 * 2, :hrp.shape[1] // 2 * 2]
        lrp = half(hrp)
        sr = apply_net(net, lrp)
        cub = np.clip(cv2.resize(lrp, (hrp.shape[1], hrp.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)
        lan = np.clip(cv2.resize(lrp, (hrp.shape[1], hrp.shape[0]), interpolation=cv2.INTER_LANCZOS4), 0, 1)
        print(f"学習に使わなかった部分 {nm.split()[-1]}: 半分の大きさから戻した結果 - バイキュービック {psnr(cub, hrp):.2f} dB、Lanczos {psnr(lan, hrp):.2f} dB、学習した拡大 {psnr(sr, hrp):.2f} dB")
        Image.fromarray((np.concatenate([hrp, lan, sr], 1) * 255).astype(np.uint8)).save(os.path.join(ROOT, "work", "sr", f"held_{nm.replace(' ', '_')}.png"))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] != "train" and os.path.isdir(args[0]):          # (sr2x.py <project> train)
        ROOT = os.path.abspath(args[0])
        NET = os.path.join(ROOT, "work", "sr", "net.pt")
        args = args[1:]
    if args and args[0] == "train":
        train(int(args[1]) if len(args) > 1 else 6000)
