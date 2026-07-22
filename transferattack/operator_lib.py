"""
Operator Library for DPSA
=========================
Aggressive image-space operators for gradient diversity (OPS-inspired, CVPR 2025).
All operators are pure PyTorch — no PIL, low GPU overhead.

Three core pools:
  _CNN_OPERATOR_POOL   — CNN-specialized (block structure, occlusion, texture)
  _FEAT_OPERATOR_POOL  — feature-safe (intensity-only, preserves spatial layout)
  _CE_OPERATOR_POOL    — global transforms (geometry, color, occlusion, frequency)
"""

import math, random
import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Basic Operators
# ═══════════════════════════════════════════════════════════════════════

def _op_identity(x):
    return x

def _op_vflip(x):
    return x.flip(dims=(2,))

def _op_hflip(x):
    return x.flip(dims=(3,))

def _op_vshift(x):
    _, _, h, _ = x.shape
    step = torch.randint(1, h, (1,)).item()
    return x.roll(step, dims=2)

def _op_hshift(x):
    _, _, _, w = x.shape
    step = torch.randint(1, w, (1,)).item()
    return x.roll(step, dims=3)


# ── Scale ────────────────────────────────────────────────────

def _op_scale05(x): return x * 0.5
def _op_scale2(x):  return x / 2.0
def _op_scale3(x):  return x / 3.0
def _op_scale4(x):  return x / 4.0
def _op_scale5(x):  return x / 5.0
def _op_scale6(x):  return x / 6.0
def _op_scale7(x):  return x / 7.0
def _op_scale8(x):  return x / 8.0


# ── Rotation (affine_grid, GPU-native) ────────────────────────

def _make_rotate_op(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    mat = torch.tensor([[c, -s, 0.], [s, c, 0.]], dtype=torch.float32)

    def _op(x):
        theta = mat.unsqueeze(0).expand(x.shape[0], 2, 3).to(device=x.device, dtype=x.dtype)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return _op

_op_rot5    = _make_rotate_op(5)
_op_rot_n5  = _make_rotate_op(-5)
_op_rot15   = _make_rotate_op(15)
_op_rot_n15 = _make_rotate_op(-15)
_op_rot30   = _make_rotate_op(30)
_op_rot_n30 = _make_rotate_op(-30)
_op_rot45   = _make_rotate_op(45)
_op_rot_n45 = _make_rotate_op(-45)
_op_rot60   = _make_rotate_op(60)
_op_rot_n60 = _make_rotate_op(-60)
_op_rot90   = _make_rotate_op(90)
_op_rot_n90 = _make_rotate_op(-90)
_op_rot180  = _make_rotate_op(180)


# ── Shear (affine_grid) ──────────────────────────────────────

def _make_shear_op(sx, sy):
    mat = torch.tensor([[1., sx, 0.], [sy, 1., 0.]], dtype=torch.float32)

    def _op(x):
        theta = mat.unsqueeze(0).expand(x.shape[0], 2, 3).to(device=x.device, dtype=x.dtype)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return _op

_op_shear_x = _make_shear_op(0.3, 0.0)
_op_shear_y = _make_shear_op(0.0, 0.3)
_op_shear_x_mild = _make_shear_op(0.15, 0.0)
_op_shear_y_mild = _make_shear_op(0.0, 0.15)
_op_shear_x_neg = _make_shear_op(-0.3, 0.0)
_op_shear_y_neg = _make_shear_op(0.0, -0.3)


# ── Color / Intensity ────────────────────────────────────────

def _op_brightness(x):
    factor = 0.5 + torch.rand(1, device=x.device).item()
    return torch.clamp(x * factor, 0, 1)

def _make_brightness_op(low, high):
    def _op(x):
        factor = low + torch.rand(1, device=x.device).item() * (high - low)
        return torch.clamp(x * factor, 0, 1)
    return _op

_op_brightness_mild = _make_brightness_op(0.75, 1.25)
_op_brightness_strong = _make_brightness_op(0.35, 1.65)

def _op_contrast(x):
    factor = 0.5 + torch.rand(1, device=x.device).item()
    mean = x.mean(dim=(2, 3), keepdim=True)
    return torch.clamp((x - mean) * factor + mean, 0, 1)

def _make_contrast_op(low, high):
    def _op(x):
        factor = low + torch.rand(1, device=x.device).item() * (high - low)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return torch.clamp((x - mean) * factor + mean, 0, 1)
    return _op

_op_contrast_mild = _make_contrast_op(0.75, 1.25)
_op_contrast_strong = _make_contrast_op(0.35, 1.65)

def _op_gaussian_noise(x):
    noise = torch.randn_like(x) * 0.03
    return torch.clamp(x + noise, 0, 1)

def _make_gaussian_noise_op(std):
    def _op(x):
        return torch.clamp(x + torch.randn_like(x) * std, 0, 1)
    return _op

_op_gaussian_noise_mild = _make_gaussian_noise_op(0.015)
_op_gaussian_noise_strong = _make_gaussian_noise_op(0.06)

def _op_salt_pepper(x):
    mask = torch.rand_like(x)
    result = x.clone()
    result[mask < 0.025] = 0
    result[mask > 0.975] = 1
    return result

def _make_salt_pepper_op(prob):
    def _op(x):
        mask = torch.rand_like(x)
        result = x.clone()
        half = prob / 2.0
        result[mask < half] = 0
        result[mask > 1.0 - half] = 1
        return result
    return _op

_op_salt_pepper_mild = _make_salt_pepper_op(0.02)
_op_salt_pepper_strong = _make_salt_pepper_op(0.10)


# ── Crop + Resize ────────────────────────────────────────────

def _op_crop_resize(x):
    B, C, H, W = x.shape
    scale = 0.70 + torch.rand(1, device=x.device).item() * 0.25
    nh, nw = max(int(H * scale), 16), max(int(W * scale), 16)
    top = torch.randint(0, H - nh + 1, (1,)).item()
    left = torch.randint(0, W - nw + 1, (1,)).item()
    cropped = x[:, :, top:top + nh, left:left + nw]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)

def _make_crop_resize_op(min_scale, max_scale):
    def _op(x):
        B, C, H, W = x.shape
        scale = min_scale + torch.rand(1, device=x.device).item() * (max_scale - min_scale)
        nh, nw = max(int(H * scale), 16), max(int(W * scale), 16)
        top = torch.randint(0, H - nh + 1, (1,), device=x.device).item()
        left = torch.randint(0, W - nw + 1, (1,), device=x.device).item()
        cropped = x[:, :, top:top + nh, left:left + nw]
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    return _op

_op_crop_resize_mild = _make_crop_resize_op(0.85, 0.98)
_op_crop_resize_strong = _make_crop_resize_op(0.55, 0.85)


# ── Fixed DIM Resize ─────────────────────────────────────────

def _make_fixed_dim(resize_rate):
    def _op(x):
        B, C, H, W = x.shape
        r = int(H * resize_rate)
        rescaled = F.interpolate(x, size=(r, r), mode='bilinear', align_corners=False)
        return F.interpolate(rescaled, size=(H, W), mode='bilinear', align_corners=False)
    return _op

_op_dim_13 = _make_fixed_dim(1.3)
_op_dim_16 = _make_fixed_dim(1.6)


# ── OPS-style DIM Resize + Pad ───────────────────────────────

def _make_ops_dim(resize_rate):
    def _op(x):
        B, C, H, W = x.shape
        resize_h = max(int(H * resize_rate), H + 1)
        resize_w = max(int(W * resize_rate), W + 1)
        rnd_h = torch.randint(low=H, high=resize_h, size=(1,), device=x.device).item()
        rnd_w = torch.randint(low=W, high=resize_w, size=(1,), device=x.device).item()
        rescaled = F.interpolate(x, size=(rnd_h, rnd_w), mode='bilinear', align_corners=False)

        h_rem, w_rem = resize_h - rnd_h, resize_w - rnd_w
        pad_top = torch.randint(low=0, high=max(h_rem, 1), size=(1,), device=x.device).item()
        pad_left = torch.randint(low=0, high=max(w_rem, 1), size=(1,), device=x.device).item()
        padded = F.pad(
            rescaled,
            [pad_left, w_rem - pad_left, pad_top, h_rem - pad_top],
            mode='constant',
            value=0,
        )
        return F.interpolate(padded, size=(H, W), mode='bilinear', align_corners=False)
    return _op

_op_ops_dim_11 = _make_ops_dim(1.1)
_op_ops_dim_13 = _make_ops_dim(1.3)
_op_ops_dim_15 = _make_ops_dim(1.5)
_op_ops_dim_17 = _make_ops_dim(1.7)
_op_ops_dim_19 = _make_ops_dim(1.9)
_op_ops_dim_21 = _make_ops_dim(2.1)
_op_ops_dim_23 = _make_ops_dim(2.3)
_op_ops_dim_25 = _make_ops_dim(2.5)
_op_ops_dim_27 = _make_ops_dim(2.7)
_op_ops_dim_29 = _make_ops_dim(2.9)


# ── Blur / Sharpen ──────────────────────────────────────────

def _op_gaussian_blur(x):
    sigma = 0.5 + torch.rand(1, device=x.device).item() * 1.5
    ks = min(int(2 * (2 * sigma) + 1) | 1, 9)
    c = torch.arange(ks, device=x.device, dtype=torch.float32)
    m = (ks - 1) / 2.0
    g = torch.exp(-(c - m) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).view(1, 1, ks, ks).repeat(3, 1, 1, 1)
    return F.conv2d(x, k, groups=3, padding=ks // 2)

def _make_gaussian_blur_op(min_sigma, max_sigma):
    def _op(x):
        sigma = min_sigma + torch.rand(1, device=x.device).item() * (max_sigma - min_sigma)
        ks = min(int(2 * (2 * sigma) + 1) | 1, 11)
        c = torch.arange(ks, device=x.device, dtype=torch.float32)
        m = (ks - 1) / 2.0
        g = torch.exp(-(c - m) ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        k = (g[:, None] * g[None, :]).view(1, 1, ks, ks).repeat(3, 1, 1, 1)
        return F.conv2d(x, k, groups=3, padding=ks // 2)
    return _op

_op_gaussian_blur_mild = _make_gaussian_blur_op(0.30, 0.80)
_op_gaussian_blur_strong = _make_gaussian_blur_op(1.50, 3.00)

def _op_sharpen(x):
    k = torch.tensor([[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]], dtype=x.dtype, device=x.device)
    k = k.repeat(3, 1, 1, 1)
    return torch.clamp(F.conv2d(x, k, groups=3, padding=1), 0, 1)

def _make_sharpen_op(amount):
    def _op(x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return torch.clamp(x + amount * (x - low), 0, 1)
    return _op

_op_sharpen_mild = _make_sharpen_op(0.50)
_op_sharpen_strong = _make_sharpen_op(1.50)


# ── Elastic Deformation ──────────────────────────────────────

def _op_elastic(x):
    B, C, H, W = x.shape
    disp = torch.randn(B, 2, H // 8, W // 8, device=x.device) * 3
    disp = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=False)
    gy, gx = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32),
                            torch.arange(W, device=x.device, dtype=torch.float32), indexing='ij')
    grid_x = (gx + disp[:, 0]) / (W - 1) * 2 - 1
    grid_y = (gy + disp[:, 1]) / (H - 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)

def _make_elastic_op(strength):
    def _op(x):
        B, C, H, W = x.shape
        disp = torch.randn(B, 2, max(H // 8, 1), max(W // 8, 1), device=x.device) * strength
        disp = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=False)
        gy, gx = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32),
                                torch.arange(W, device=x.device, dtype=torch.float32), indexing='ij')
        grid_x = (gx + disp[:, 0]) / (W - 1) * 2 - 1
        grid_y = (gy + disp[:, 1]) / (H - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    return _op

_op_elastic_mild = _make_elastic_op(1.5)
_op_elastic_strong = _make_elastic_op(5.0)


# ── Perspective Transform ────────────────────────────────────

def _solve_perspective(src, dst):
    A = torch.cat([src[:3], torch.ones(3, 1, device=src.device)], dim=1)
    return torch.linalg.solve(A, dst[:3]).t()

def _op_perspective(x):
    B, _, H, W = x.shape
    d = 0.1
    src = torch.tensor([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=torch.float32, device=x.device)
    dst = src + torch.rand(4, 2, device=x.device) * 2 * d - d
    theta = _solve_perspective(src, dst).to(x.device).to(x.dtype)
    grid = F.affine_grid(theta.unsqueeze(0).expand(B, 2, 3), x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

def _make_perspective_op(d):
    def _op(x):
        B, _, H, W = x.shape
        src = torch.tensor([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=torch.float32, device=x.device)
        dst = src + torch.rand(4, 2, device=x.device) * 2 * d - d
        theta = _solve_perspective(src, dst).to(x.device).to(x.dtype)
        grid = F.affine_grid(theta.unsqueeze(0).expand(B, 2, 3), x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return _op

_op_perspective_mild = _make_perspective_op(0.05)
_op_perspective_strong = _make_perspective_op(0.15)


# ── Random Erase ─────────────────────────────────────────────

def _op_random_erase(x):
    B, C, H, W = x.shape
    result = x.clone()
    for b in range(B):
        area = 0.20 + torch.rand(1, device=x.device).item() * 0.30
        aspect = 0.5 + torch.rand(1, device=x.device).item() * 1.5
        eh = min(int(H * (area * aspect) ** 0.5), H)
        ew = min(int(W * (area / aspect) ** 0.5), W)
        top = torch.randint(0, max(H - eh + 1, 1), (1,)).item()
        left = torch.randint(0, max(W - ew + 1, 1), (1,)).item()
        result[b, :, top:top + eh, left:left + ew] = 0
    return result

def _make_random_erase_op(min_area, max_area):
    def _op(x):
        B, C, H, W = x.shape
        result = x.clone()
        for b in range(B):
            area = min_area + torch.rand(1, device=x.device).item() * (max_area - min_area)
            aspect = 0.5 + torch.rand(1, device=x.device).item() * 1.5
            eh = min(int(H * (area * aspect) ** 0.5), H)
            ew = min(int(W * (area / aspect) ** 0.5), W)
            top = torch.randint(0, max(H - eh + 1, 1), (1,), device=x.device).item()
            left = torch.randint(0, max(W - ew + 1, 1), (1,), device=x.device).item()
            result[b, :, top:top + eh, left:left + ew] = 0
        return result
    return _op

_op_random_erase_small = _make_random_erase_op(0.05, 0.18)
_op_random_erase_large = _make_random_erase_op(0.35, 0.60)


# ── Block Shuffle (BSR, ECCV'22) ─────────────────────────────

def _op_bsr_shuffle(x):
    B, C, H, W = x.shape
    n_block = 3
    result = x.clone()
    for b in range(B):
        rand = np.random.uniform(2, size=n_block)
        rand_norm = np.round(rand / rand.sum() * H).astype(np.int32)
        rand_norm[rand_norm.argmax()] += H - rand_norm.sum()
        strips_h = list(torch.split(x[b:b + 1], tuple(rand_norm), dim=2))
        random.shuffle(strips_h)
        for i in range(len(strips_h)):
            strip = strips_h[i]
            deg = random.uniform(-24, 24)
            rad = math.radians(deg)
            c, s_c = math.cos(rad), math.sin(rad)
            mat = torch.tensor([[c, -s_c, 0.], [s_c, c, 0.]], dtype=x.dtype, device=x.device)
            theta = mat.unsqueeze(0)
            grid = F.affine_grid(theta, strip.shape, align_corners=False)
            strips_h[i] = F.grid_sample(strip, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        final_strips = []
        for strip in strips_h:
            rand_w = np.random.uniform(2, size=n_block)
            rand_norm_w = np.round(rand_w / rand_w.sum() * W).astype(np.int32)
            rand_norm_w[rand_norm_w.argmax()] += W - rand_norm_w.sum()
            sub = list(torch.split(strip, tuple(rand_norm_w), dim=3))
            random.shuffle(sub)
            final_strips.append(torch.cat(sub, dim=3))
        result[b:b + 1] = torch.cat(final_strips, dim=2)
    return result

def _make_bsr_shuffle_op(n_block, max_deg):
    def _op(x):
        B, C, H, W = x.shape
        result = x.clone()
        for b in range(B):
            rand = np.random.uniform(2, size=n_block)
            rand_norm = np.round(rand / rand.sum() * H).astype(np.int32)
            rand_norm[rand_norm.argmax()] += H - rand_norm.sum()
            strips_h = list(torch.split(x[b:b + 1], tuple(rand_norm), dim=2))
            random.shuffle(strips_h)
            for i in range(len(strips_h)):
                strip = strips_h[i]
                deg = random.uniform(-max_deg, max_deg)
                rad = math.radians(deg)
                c, s_c = math.cos(rad), math.sin(rad)
                mat = torch.tensor([[c, -s_c, 0.], [s_c, c, 0.]], dtype=x.dtype, device=x.device)
                theta = mat.unsqueeze(0)
                grid = F.affine_grid(theta, strip.shape, align_corners=False)
                strips_h[i] = F.grid_sample(strip, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
            final_strips = []
            for strip in strips_h:
                rand_w = np.random.uniform(2, size=n_block)
                rand_norm_w = np.round(rand_w / rand_w.sum() * W).astype(np.int32)
                rand_norm_w[rand_norm_w.argmax()] += W - rand_norm_w.sum()
                sub = list(torch.split(strip, tuple(rand_norm_w), dim=3))
                random.shuffle(sub)
                final_strips.append(torch.cat(sub, dim=3))
            result[b:b + 1] = torch.cat(final_strips, dim=2)
        return result
    return _op

_op_bsr_shuffle_coarse = _make_bsr_shuffle_op(2, 16)
_op_bsr_shuffle_fine = _make_bsr_shuffle_op(4, 28)


# ── Block Operator (SIA-inspired) ────────────────────────────

def _op_block_operator(x):
    B, C, H, W = x.shape
    n_blocks = random.randint(2, 4)
    result = x.clone()
    ops = [lambda t: t.flip(2), lambda t: t.flip(3),
           lambda t: t.roll(random.randint(1, 16), 2),
           lambda t: t * random.uniform(0.3, 0.9),
           lambda t: t.rot90(k=2, dims=(2, 3)),
           lambda t: torch.clamp(t + torch.randn_like(t) * 0.05, 0, 1)]
    y_pts = sorted([0] + [random.randint(H // 4, 3 * H // 4) for _ in range(n_blocks - 1)] + [H])
    x_pts = sorted([0] + [random.randint(W // 4, 3 * W // 4) for _ in range(n_blocks - 1)] + [W])
    for i in range(n_blocks):
        for j in range(n_blocks):
            op = random.choice(ops)
            result[:, :, y_pts[i]:y_pts[i + 1], x_pts[j]:x_pts[j + 1]] = op(
                x[:, :, y_pts[i]:y_pts[i + 1], x_pts[j]:x_pts[j + 1]])
    return result

def _make_block_operator_op(min_blocks, max_blocks, noise_std):
    def _op(x):
        B, C, H, W = x.shape
        n_blocks = random.randint(min_blocks, max_blocks)
        result = x.clone()
        ops = [lambda t: t.flip(2), lambda t: t.flip(3),
               lambda t: t.roll(random.randint(1, 16), 2),
               lambda t: t * random.uniform(0.3, 0.9),
               lambda t: t.rot90(k=2, dims=(2, 3)),
               lambda t: torch.clamp(t + torch.randn_like(t) * noise_std, 0, 1)]
        y_pts = sorted([0] + [random.randint(H // 4, 3 * H // 4) for _ in range(n_blocks - 1)] + [H])
        x_pts = sorted([0] + [random.randint(W // 4, 3 * W // 4) for _ in range(n_blocks - 1)] + [W])
        for i in range(n_blocks):
            for j in range(n_blocks):
                op = random.choice(ops)
                result[:, :, y_pts[i]:y_pts[i + 1], x_pts[j]:x_pts[j + 1]] = op(
                    x[:, :, y_pts[i]:y_pts[i + 1], x_pts[j]:x_pts[j + 1]])
        return result
    return _op

_op_block_operator_coarse = _make_block_operator_op(2, 3, 0.03)
_op_block_operator_fine = _make_block_operator_op(4, 5, 0.06)


# ── MaskBlock ────────────────────────────────────────────────

def _op_maskblock(x):
    B, C, H, W = x.shape
    result = x.clone()
    bh = random.randint(H // 5, H // 3)
    bw = random.randint(W // 5, W // 3)
    top = random.randint(0, max(H - bh, 1))
    left = random.randint(0, max(W - bw, 1))
    result[:, :, top:top + bh, left:left + bw] = 0
    return result

def _make_maskblock_op(min_frac, max_frac):
    def _op(x):
        B, C, H, W = x.shape
        result = x.clone()
        frac = min_frac + random.random() * (max_frac - min_frac)
        bh = max(int(H * frac), 1)
        bw = max(int(W * frac), 1)
        top = random.randint(0, max(H - bh, 1))
        left = random.randint(0, max(W - bw, 1))
        result[:, :, top:top + bh, left:left + bw] = 0
        return result
    return _op

_op_maskblock_small = _make_maskblock_op(0.10, 0.20)
_op_maskblock_large = _make_maskblock_op(0.30, 0.45)

def _op_maskblock_grid(x):
    B, C, H, W = x.shape
    patch = max(min(H, W) // 4, 1)
    n_h, n_w = max(H // patch, 1), max(W // patch, 1)
    result = x.clone()
    for b in range(B):
        i, j = random.randint(0, n_h - 1), random.randint(0, n_w - 1)
        y1, y2 = i * patch, min((i + 1) * patch, H)
        x1, x2 = j * patch, min((j + 1) * patch, W)
        result[b, :, y1:y2, x1:x2] = 0
    return result

def _make_maskblock_grid_op(divisor):
    def _op(x):
        B, C, H, W = x.shape
        patch = max(min(H, W) // divisor, 1)
        n_h, n_w = max(H // patch, 1), max(W // patch, 1)
        result = x.clone()
        for b in range(B):
            i, j = random.randint(0, n_h - 1), random.randint(0, n_w - 1)
            y1, y2 = i * patch, min((i + 1) * patch, H)
            x1, x2 = j * patch, min((j + 1) * patch, W)
            result[b, :, y1:y2, x1:x2] = 0
        return result
    return _op

_op_maskblock_grid_small = _make_maskblock_grid_op(6)
_op_maskblock_grid_large = _make_maskblock_grid_op(3)


# ── Affine Translate ─────────────────────────────────────────

def _op_affine_translate(x):
    B, C, H, W = x.shape
    tx = random.random() * 0.4 - 0.2
    ty = random.random() * 0.4 - 0.2
    mat = torch.tensor([[1., 0., tx], [0., 1., ty]], dtype=x.dtype, device=x.device)
    theta = mat.unsqueeze(0).expand(B, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

def _make_affine_translate_op(max_shift):
    def _op(x):
        B, C, H, W = x.shape
        tx = random.random() * 2 * max_shift - max_shift
        ty = random.random() * 2 * max_shift - max_shift
        mat = torch.tensor([[1., 0., tx], [0., 1., ty]], dtype=x.dtype, device=x.device)
        theta = mat.unsqueeze(0).expand(B, 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return _op

_op_affine_translate_mild = _make_affine_translate_op(0.10)
_op_affine_translate_strong = _make_affine_translate_op(0.30)


# ── Frequency High Attenuation (MFI/SSM-inspired) ────────────

def _op_freq_high_attenuate(x):
    B, C, H, W = x.shape
    X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )
    cy, cx = H // 2, W // 2
    dist = torch.sqrt((yy - cy).float().pow(2) + (xx - cx).float().pow(2))
    radius = min(H, W) * random.uniform(0.18, 0.35)
    low = torch.sigmoid(-(dist - radius) / 8.0).view(1, 1, H, W)
    high_gain = random.uniform(0.25, 0.75)
    X = X * (low + high_gain * (1.0 - low))
    out = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), dim=(-2, -1)).real
    return torch.clamp(out, 0, 1)

def _op_freq_high_boost(x):
    B, C, H, W = x.shape
    X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )
    cy, cx = H // 2, W // 2
    dist = torch.sqrt((yy - cy).float().pow(2) + (xx - cx).float().pow(2))
    radius = min(H, W) * random.uniform(0.18, 0.35)
    low = torch.sigmoid(-(dist - radius) / 8.0).view(1, 1, H, W)
    high_gain = random.uniform(1.15, 1.75)
    X = X * (low + high_gain * (1.0 - low))
    out = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), dim=(-2, -1)).real
    return torch.clamp(out, 0, 1)

def _op_freq_mid_attenuate(x):
    B, C, H, W = x.shape
    X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )
    cy, cx = H // 2, W // 2
    dist = torch.sqrt((yy - cy).float().pow(2) + (xx - cx).float().pow(2))
    center = min(H, W) * random.uniform(0.18, 0.32)
    width = min(H, W) * random.uniform(0.05, 0.10)
    band = torch.exp(-((dist - center) ** 2) / (2 * width ** 2)).view(1, 1, H, W)
    gain = random.uniform(0.35, 0.70)
    X = X * (1.0 - band + gain * band)
    out = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), dim=(-2, -1)).real
    return torch.clamp(out, 0, 1)


# ═══════════════════════════════════════════════════════════════════════
# Three Core Operator Pools
# ═══════════════════════════════════════════════════════════════════════

# OPS original operator set: flips/shifts, rotations, scaling, and DIM.
_OPS_OPERATOR_POOL = [
    _op_identity,
    _op_vflip, _op_hflip,
    _op_vshift, _op_hshift,
    _op_rot5, _op_rot_n5,
    _op_rot15, _op_rot_n15,
    _op_rot45, _op_rot_n45,
    _op_rot90, _op_rot_n90,
    _op_rot180,
    _op_scale2, _op_scale3, _op_scale4, _op_scale5,
    _op_scale6, _op_scale7, _op_scale8,
    _op_ops_dim_11, _op_ops_dim_13, _op_ops_dim_15, _op_ops_dim_17, _op_ops_dim_19,
    _op_ops_dim_21, _op_ops_dim_23, _op_ops_dim_25, _op_ops_dim_27, _op_ops_dim_29,
]

# Basic pool without high-level special operators such as BSR, block masking,
# perspective/elastic deformation, random erasing, and frequency transforms.
_BASIC_OPERATOR_POOL = [
    _op_identity,
    _op_vflip, _op_hflip,
    _op_vshift, _op_hshift,
    _op_affine_translate, _op_affine_translate_mild,
    _op_rot5, _op_rot_n5,
    _op_rot15, _op_rot_n15,
    _op_rot30, _op_rot_n30,
    _op_rot45, _op_rot_n45,
    _op_rot90, _op_rot_n90,
    _op_rot180,
    _op_shear_x_mild, _op_shear_y_mild,
    _op_crop_resize, _op_crop_resize_mild,
    _op_dim_13, _op_dim_16,
    _op_ops_dim_11, _op_ops_dim_13, _op_ops_dim_15, _op_ops_dim_17, _op_ops_dim_19,
    _op_ops_dim_21, _op_ops_dim_23, _op_ops_dim_25, _op_ops_dim_27, _op_ops_dim_29,
    _op_scale05,
    _op_scale2, _op_scale3, _op_scale4,
    _op_scale5, _op_scale6, _op_scale7, _op_scale8,
    _op_brightness, _op_brightness_mild, _op_brightness_strong,
    _op_contrast, _op_contrast_mild, _op_contrast_strong,
    _op_gaussian_noise, _op_gaussian_noise_mild, _op_gaussian_noise_strong,
    _op_salt_pepper, _op_salt_pepper_mild,
    _op_gaussian_blur, _op_gaussian_blur_mild, _op_gaussian_blur_strong,
    _op_sharpen, _op_sharpen_mild, _op_sharpen_strong,
]

# Pool A — CNN-specialized: block structure, local occlusion, texture
# Unified pool: all operators from the three pools, deduplicated
_UNIFIED_OPERATOR_POOL = []

_CNN_OPERATOR_POOL = [
    _op_identity,
    _op_bsr_shuffle, _op_bsr_shuffle_coarse, _op_bsr_shuffle_fine,
    _op_block_operator, _op_block_operator_coarse, _op_block_operator_fine,
    _op_maskblock_grid, _op_maskblock_grid_small, _op_maskblock_grid_large,
    _op_maskblock, _op_maskblock_small, _op_maskblock_large,
    _op_crop_resize, _op_crop_resize_mild, _op_crop_resize_strong,
    _op_affine_translate, _op_affine_translate_mild, _op_affine_translate_strong,
    _op_rot5,
    _op_rot_n5,
    _op_gaussian_blur, _op_gaussian_blur_mild, _op_gaussian_blur_strong,
    _op_sharpen, _op_sharpen_mild, _op_sharpen_strong,
    _op_freq_high_attenuate, _op_freq_high_boost, _op_freq_mid_attenuate,
]

# Pool B — Feature-safe: intensity-only transforms, preserves spatial layout
_FEAT_OPERATOR_POOL = [
    _op_identity,
    _op_scale05,
    _op_scale2, _op_scale3, _op_scale4,
    _op_scale5, _op_scale6, _op_scale7, _op_scale8,
    _op_brightness, _op_brightness_mild, _op_brightness_strong,
    _op_contrast, _op_contrast_mild, _op_contrast_strong,
    _op_gaussian_noise, _op_gaussian_noise_mild, _op_gaussian_noise_strong,
]

# Pool C — Global: full-image geometry, color, occlusion, frequency transforms
_CE_OPERATOR_POOL = [
    _op_identity,
    # mild geometry
    _op_vflip, _op_hflip,
    _op_vshift, _op_hshift,
    _op_affine_translate,
    _op_rot5, _op_rot_n5,
    _op_rot15, _op_rot_n15,
    _op_rot30, _op_rot_n30,
    _op_rot45, _op_rot_n45,
    _op_rot60, _op_rot_n60,
    _op_rot90, _op_rot_n90, _op_rot180,
    _op_shear_x, _op_shear_y,
    _op_shear_x_mild, _op_shear_y_mild, _op_shear_x_neg, _op_shear_y_neg,
    _op_crop_resize, _op_crop_resize_mild, _op_crop_resize_strong,
    _op_dim_13, _op_dim_16,
    _op_ops_dim_11, _op_ops_dim_13, _op_ops_dim_15, _op_ops_dim_17, _op_ops_dim_19,
    _op_ops_dim_21, _op_ops_dim_23, _op_ops_dim_25, _op_ops_dim_27, _op_ops_dim_29,
    _op_perspective, _op_perspective_mild, _op_perspective_strong,
    _op_elastic, _op_elastic_mild, _op_elastic_strong,
    # intensity / texture
    _op_brightness, _op_brightness_mild, _op_brightness_strong,
    _op_contrast, _op_contrast_mild, _op_contrast_strong,
    _op_gaussian_noise, _op_gaussian_noise_mild, _op_gaussian_noise_strong,
    _op_salt_pepper, _op_salt_pepper_mild, _op_salt_pepper_strong,
    _op_gaussian_blur, _op_gaussian_blur_mild, _op_gaussian_blur_strong,
    _op_sharpen, _op_sharpen_mild, _op_sharpen_strong,
    _op_freq_high_attenuate, _op_freq_high_boost, _op_freq_mid_attenuate,
    # occlusion / block structure
    _op_maskblock, _op_maskblock_small, _op_maskblock_large,
    _op_maskblock_grid, _op_maskblock_grid_small, _op_maskblock_grid_large,
    _op_random_erase, _op_random_erase_small, _op_random_erase_large,
    _op_block_operator, _op_block_operator_coarse, _op_block_operator_fine,
    _op_bsr_shuffle, _op_bsr_shuffle_coarse, _op_bsr_shuffle_fine,
]

# Build unified pool (deduplicated, preserve order)
_seen = set()
_UNIFIED_OPERATOR_POOL = []
for _op in _CNN_OPERATOR_POOL + _FEAT_OPERATOR_POOL + _CE_OPERATOR_POOL:
    if _op not in _seen:
        _UNIFIED_OPERATOR_POOL.append(_op)
        _seen.add(_op)
del _seen, _op
