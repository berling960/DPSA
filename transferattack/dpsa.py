"""
DPSA: DINOv3-Guided Path Sampling Attack.

Four integrated components:
  1. Operator path sampling → gradient diversity
  2. GDS (Gradient Distribution Synthesis) → cross-view fusion
  3. OPS-style neighborhood paths under a linear path budget
  4. Forward-only local probe → endpoint quality estimate

Usage:  attacker = DPSA('dinov3', epoch=15, num_op_samples=30, ...)
        attacker = DPSA('resnet50', epoch=15, num_op_samples=30, ...)
        perturbation = attacker(images, labels)
"""

import os, random
import torch, torch.nn as nn, torch.nn.functional as F
from .utils import *
from .utils import (
    _dataset_checkpoint_path,
    _head_state_from_checkpoint,
    _num_classes_for_dataset,
)
from .attack import Attack
from .operator_lib import (
    _BASIC_OPERATOR_POOL,
    _OPS_OPERATOR_POOL,
    _UNIFIED_OPERATOR_POOL,
    _op_identity,
)


class CUDAPreprocessing(nn.Module):
    """CUDA-safe resize + normalize."""
    def __init__(self, size, mean, std):
        super().__init__()
        self.size = size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        return (x - self.mean) / self.std


class DINOv3AttackModel(nn.Module):
    """DINOv3 wrapper for attack logits."""
    def __init__(self, backbone, head, preprocess):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.preprocess = preprocess

    def forward(self, x):
        x = self.preprocess(x)
        return self.head(self.backbone(x).pooler_output)


class DPSA(Attack):
    """DPSA: DINOv3-Guided Path Sampling Attack."""

    def __init__(self, model_name,
                 epsilon=16 / 255, alpha=1.6 / 255, epoch=15, decay=1.0,
                 targeted=False, random_start=False, norm='linfty',
                 loss='crossentropy', device=None,
                 num_op_samples=30, pool_chain_length=3, num_neighbor=15,
                 operator_pool_variant='full',
                 use_diversity=True, diversity_prob=1.0,
                 gds_diversity_weight=0.25, gds_conflict_floor=0.05,
                 gds_loss_weight=0.3,
                 local_probe_samples=1, local_probe_radius=0.05,
                 local_probe_weight=0.25, local_probe_std_weight=0.00,
                 dataset_name='ImageNet',
                 pretrained_models_root='./pretrained_models',
                 **kwargs):
        self.num_op_samples = num_op_samples
        self.pool_chain_length = max(int(pool_chain_length), 1)
        self.num_neighbor = max(int(num_neighbor), 0)
        self.operator_pool_variant = operator_pool_variant.lower()
        self.operator_pool = self._select_operator_pool(self.operator_pool_variant)
        self.use_diversity = use_diversity
        self.diversity_prob = diversity_prob
        self.gds_diversity_weight = min(max(float(gds_diversity_weight), 0.0), 0.8)
        self.gds_conflict_floor = min(max(float(gds_conflict_floor), 0.0), 0.5)
        self.gds_loss_weight = min(max(float(gds_loss_weight), 0.0), 2.0)
        self.local_probe_samples = max(int(local_probe_samples), 0)
        self.local_probe_radius = max(float(local_probe_radius), 0.0)
        self.local_probe_weight = min(max(float(local_probe_weight), 0.0), 2.0)
        self.local_probe_std_weight = min(max(float(local_probe_std_weight), 0.0), 2.0)
        self.dataset_name = dataset_name
        self.pretrained_models_root = pretrained_models_root
        super().__init__('DPSA', model_name, epsilon, targeted, random_start,
                         norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay

    # ── Model Loading ──────────────────────────────────────────

    def load_model(self, model_name):
        if model_name.lower().startswith('dinov3'):
            return self._build_dinov3_model(model_name)
        return self._build_generic_model(model_name)

    def _build_dinov3_model(self, model_name):
        from transformers import AutoConfig, AutoModel
        hf_ids = {
            'dinov3': 'facebook/dinov3-small', 'dinov3-vits16': 'facebook/dinov3-small',
            'dinov3-vitb16': 'facebook/dinov3-base', 'dinov3-vitl16': 'facebook/dinov3-large',
        }
        model_id = hf_ids.get(model_name.lower(), 'facebook/dinov3-small')
        local = os.path.join(self.pretrained_models_root, model_id.split('/')[-1])
        model_path = local if os.path.exists(local) else model_id
        config = AutoConfig.from_pretrained(model_path)
        backbone = AutoModel.from_pretrained(model_path, config=config)
        if self.dataset_name.lower() == 'imagenet':
            head = nn.Linear(config.hidden_size, 1000)
            head_path = os.path.join(self.pretrained_models_root, 'dinov3-vits16-5ep-head.pth')
            if os.path.exists(head_path):
                head.load_state_dict(torch.load(head_path, map_location='cpu'))
        else:
            num_classes = _num_classes_for_dataset(self.dataset_name)
            head = nn.Linear(config.hidden_size, num_classes)
            ckpt_path = _dataset_checkpoint_path(
                self.pretrained_models_root, self.dataset_name, 'dinov3-vits16'
            )
            if not ckpt_path.exists():
                raise FileNotFoundError(f"DINOv3 CUB head not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            head.load_state_dict(_head_state_from_checkpoint(ckpt))
        preprocess = CUDAPreprocessing(224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        model = DINOv3AttackModel(backbone, head, preprocess)
        for p in model.parameters():
            p.requires_grad = False
        return model.eval().cuda()

    def _build_generic_model(self, model_name):
        model = load_single_model(model_name, dataset_name=self.dataset_name,
                                  pretrained_models_root=self.pretrained_models_root)
        for p in model.parameters():
            p.requires_grad = False
        return model.eval().cuda()

    # ── Input Diversity (DIM) ─────────────────────────────────

    def transform(self, data, **kwargs):
        if not self.use_diversity:
            return data
        if torch.rand(1).item() > self.diversity_prob:
            return data
        B, C, H, W = data.shape
        factor = 0.85 + torch.rand(1).item() * 0.15
        nh, nw = max(int(H * factor), 2), max(int(W * factor), 2)
        resized = F.interpolate(data, size=(nh, nw), mode='bilinear', align_corners=False)
        ph, pw = H - nh, W - nw
        pt = torch.randint(0, ph + 1, (1,)).item()
        pl = torch.randint(0, pw + 1, (1,)).item()
        return F.pad(resized, [pl, pw - pl, pt, ph - pt], mode='constant', value=0)

    # ── Operator Pool & Chaining ───────────────────────────────

    @staticmethod
    def _select_operator_pool(variant):
        pools = {
            'full': _UNIFIED_OPERATOR_POOL,
            'unified': _UNIFIED_OPERATOR_POOL,
            'ops': _OPS_OPERATOR_POOL,
            'basic': _BASIC_OPERATOR_POOL,
            'no_special': _BASIC_OPERATOR_POOL,
        }
        if variant not in pools:
            raise ValueError(f'Unknown operator_pool_variant: {variant}')
        return pools[variant]

    def _sample_chained_ops(self):
        ops = []
        for _ in range(self.pool_chain_length):
            candidates = [op for op in self.operator_pool if op is not _op_identity]
            ops.append(random.choice(candidates))
        return ops

    @staticmethod
    def _apply_route_ops(x, route_ops):
        for op in route_ops:
            x = op(x)
        return x

    # ── Loss Functions ─────────────────────────────────────────

    def get_loss(self, logits, label):
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)

    # ── Gradient Utilities ─────────────────────────────────────

    @staticmethod
    def _normalize_grad(grad, eps=1e-12):
        B = grad.shape[0]
        return grad / (grad.detach().abs().reshape(B, -1).mean(dim=1).reshape(B, 1, 1, 1) + eps)

    # ── GDS ─────────────────────────────────────────────────────

    def _gds_fuse(self, grad_views, loss_values=None, eps=1e-12):
        if len(grad_views) == 1:
            return grad_views[0]
        raw = torch.stack(grad_views, dim=0)
        base_scale = raw.mean(dim=0).detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        norm = torch.stack([self._normalize_grad(g) for g in grad_views], dim=0)
        mean = norm.mean(dim=0)
        vf, mf = norm.flatten(2), mean.flatten(1)
        cos = (vf * mf.unsqueeze(0)).sum(dim=2) / (vf.norm(dim=2).clamp_min(eps) * mf.norm(dim=1).clamp_min(eps).unsqueeze(0))
        quality = torch.zeros(len(grad_views), 1, device=norm.device, dtype=norm.dtype)
        if loss_values is not None and len(loss_values) == len(grad_views):
            losses = torch.stack([v.detach().to(device=norm.device, dtype=norm.dtype).reshape(()) for v in loss_values])
            if losses.numel() > 1:
                losses = (losses - losses.mean()) / (losses.std(unbiased=False) + eps)
                quality = torch.sigmoid(losses).view(-1, 1)
        floor = self.gds_conflict_floor / max(len(grad_views) ** 0.5, 1.0)
        pq = 1.0 + self.gds_loss_weight * quality
        sw = (torch.relu(cos).pow(2.0) + floor) * pq
        sw = sw / sw.sum(dim=0, keepdim=True).clamp_min(eps)
        stable = (norm * sw[:, :, None, None, None]).sum(dim=0)
        stable = stable * (0.75 + 0.25 * norm.sign().mean(dim=0).abs())
        stable = self._normalize_grad(stable)
        residual = norm - stable.unsqueeze(0)
        rw = ((1.0 - cos.clamp(-1.0, 1.0)) * torch.relu(cos) + floor) * pq
        rw = rw / rw.sum(dim=0, keepdim=True).clamp_min(eps)
        diversity = (residual * rw[:, :, None, None, None]).sum(dim=0)
        B = diversity.shape[0]
        df, sf = diversity.reshape(B, -1), stable.reshape(B, -1)
        dot = (df * sf).sum(dim=1, keepdim=True)
        s_n2 = (sf * sf).sum(dim=1, keepdim=True).clamp_min(eps)
        df = torch.where(dot < 0, df - dot / s_n2 * sf, df)
        diversity = self._normalize_grad(df.reshape_as(diversity))
        return ((1.0 - self.gds_diversity_weight) * stable + self.gds_diversity_weight * diversity) * base_scale

    # ── Main Attack Loop ───────────────────────────────────────

    def _forward_gds_random5(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        grad_momentum = 0

        for step_idx in range(self.epoch):
            x_adv = data + delta
            x_div = self.transform(x_adv, grad_momentum=grad_momentum)

            def _path_loss_grad(x_in):
                logits = self.model(x_in)
                loss = self.get_loss(logits, label)
                return loss, logits

            def _path_loss_value(x_in):
                return self.get_loss(self.model(x_in), label).detach()

            main_loss, logits = _path_loss_grad(x_div)
            main_grad = torch.autograd.grad(main_loss, delta, retain_graph=False, create_graph=False)[0]
            grad_views = [main_grad]
            loss_values = [self.get_loss(logits, label).detach()]

            num_cross = min(self.num_neighbor, max(self.num_op_samples, 0))
            for path_idx in range(max(self.num_op_samples, 0)):
                route_ops = self._sample_chained_ops()
                if path_idx >= self.num_op_samples - num_cross:
                    # OPS-style radii, kept moderate to preserve DPSA's linear path budget.
                    radii = [0.50, 1.00, 1.50, 2.00]
                    r = radii[(path_idx - (self.num_op_samples - num_cross)) % len(radii)]
                    n = torch.empty_like(delta).uniform_(-r * self.epsilon, r * self.epsilon)
                    x_in = torch.clamp(x_adv + n, img_min, img_max)
                else:
                    x_in = x_adv
                x_view = self._apply_route_ops(x_in, route_ops)
                path_loss, _ = _path_loss_grad(x_view)
                op_grad = torch.autograd.grad(path_loss, delta, retain_graph=False, create_graph=False)[0]
                quality = path_loss.detach()
                if self.local_probe_samples > 0 and self.local_probe_radius > 0:
                    with torch.no_grad():
                        probe_losses = []
                        probe_radius = self.local_probe_radius * self.epsilon
                        center = x_view.detach()
                        for _ in range(self.local_probe_samples):
                            probe_noise = torch.empty_like(center).uniform_(-probe_radius, probe_radius)
                            probe_x = torch.clamp(center + probe_noise, img_min, img_max)
                            probe_losses.append(_path_loss_value(probe_x))
                        probes = torch.stack([p.reshape(()) for p in probe_losses])
                        quality = quality + self.local_probe_weight * probes.mean()
                        if probes.numel() > 1:
                            quality = quality - self.local_probe_std_weight * probes.std(unbiased=False)
                loss_values.append(quality)
                grad_views.append(op_grad)

            total_grad = self._gds_fuse(grad_views, loss_values)
            grad_momentum = self.get_momentum(total_grad, grad_momentum)
            delta = self.update_delta(delta, data, grad_momentum, self.alpha)

        return delta.detach()

    def forward(self, data, label, **kwargs):
        return self._forward_gds_random5(data, label, **kwargs)


LPSA = DPSA
