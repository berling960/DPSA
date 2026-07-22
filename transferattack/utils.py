import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import pandas as pd
import timm
import os
from pathlib import Path

img_height, img_width = 224, 224
img_max, img_min = 1., 0

cnn_model_paper = ['resnet50', 'vgg16', 'mobilenet_v2', 'inception_v3']
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224',
                   'visformer_small', 'swin_tiny_patch4_window7_224']

cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                      'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]



def _load_pretrained_model_generic(model_name, pretrained_models_root='./pretrained_models'):
    """
    Generic model loader: DINOv3/DINOv2/MAE → Mamba → torchvision → timm.
    Returns raw model (no preprocessing wrapper).
    """
    name_lower = model_name.lower()
    if name_lower.startswith('dinov3'):
        return _load_dinov3_model_raw(model_name, pretrained_models_root)
    if name_lower.startswith('dinov2') or name_lower.startswith('mae'):
        return _load_selfsup_model_raw(model_name)

    # Mamba/SSM models with local weights (expected in pretrained_models/)
    mamba_local_map = {
        'mambaout_small_rw': os.path.join(pretrained_models_root, 'mambaout_small', 'pytorch_model.bin'),
        'mambaout_kobe': os.path.join(pretrained_models_root, 'mambaout_kobe', 'pytorch_model.bin'),
        'vmamba_small': os.path.join(pretrained_models_root, 'vmamba_small'),
    }
    if model_name in mamba_local_map:
        local_path = mamba_local_map[model_name]
        if 'vmamba' in model_name:
            sys.path.insert(0, local_path)
            try:
                from configuration_vmamba import VMambaConfig
                from modeling_vmamba import VMambaForImageClassification
                config = VMambaConfig.from_pretrained(local_path)
                hf_model = VMambaForImageClassification.from_pretrained(local_path, config=config)
            finally:
                sys.path.pop(0)
            class VMambaWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x): return self.m(x).logits
            return VMambaWrapper(hf_model)
        # MambaOut: timm-based, load local state dict
        model = timm.create_model(model_name, pretrained=False)
        model.load_state_dict(torch.load(local_path, map_location='cpu'), strict=False)
        return model

    # Standard torchvision / timm models (auto-download pretrained weights)
    if model_name in models.__dict__:
        print(f"=> Loading {model_name} from torchvision (ImageNet pretrained)")
        return models.__dict__[model_name](weights="DEFAULT")
    elif model_name in timm.list_models():
        print(f"=> Loading {model_name} from timm (ImageNet pretrained)")
        return timm.create_model(model_name, pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not found in torchvision or timm")


def get_model_architecture(model_name, num_classes):
    """
    Get model architecture with specified number of classes
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
    
    Returns:
        model: PyTorch model
    """
    if model_name == 'vmamba_small':
        return _build_vmamba_classifier(num_classes, pretrained=False)

    # Try torchvision models first
    if hasattr(models, model_name):
        model = getattr(models, model_name)(weights=None)
        
        # Modify final layer based on architecture
        if 'resnet' in model_name or 'resnext' in model_name or 'regnet' in model_name:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif 'densenet' in model_name:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif 'vgg' in model_name or 'alexnet' in model_name:
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)
        elif 'inception' in model_name:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif 'mobilenet' in model_name:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {model_name}")
    
    # Try timm models
    else:
        try:
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        except Exception as e:
            raise ValueError(f"Model {model_name} not found: {e}")
    
    return model

def get_model_display_name(model_name):
    """
    Get human-readable display name with parameter count for DINOv3 models.

    Args:
        model_name: raw model name from CLI (e.g., 'dinov3', 'dinov3-vitb16')

    Returns:
        display name with size info (e.g., 'dinov3_small', 'dinov3_base', 'dinov3_large')
    """
    dinov3_display_map = {
        'dinov3':          'dinov3_small',     # ViT-Small/16: 21.6M backbone
        'dinov3-vits16':   'dinov3_small',     # ViT-Small/16: 21.6M backbone
        'dinov3-vitb16':   'dinov3_base',      # ViT-Base/16:  85.7M backbone
        'dinov3-vitl16':   'dinov3_large',     # ViT-Large/16: 303.1M backbone
    }
    if model_name.lower() in dinov3_display_map:
        return dinov3_display_map[model_name.lower()]
    return model_name


def _canonical_dataset_name(dataset_name):
    return 'CUB' if str(dataset_name).upper() == 'CUB1000' else dataset_name


def _num_classes_for_dataset(dataset_name):
    dataset_name = _canonical_dataset_name(dataset_name)
    return {
        'UCM': 21,
        'AID': 30,
        'NWPU': 45,
        'RSSCN7': 7,
        'CUB': 200,
        'AIR': 100,
        'STCAR': 196,
    }[dataset_name]


def _dataset_checkpoint_path(pretrained_models_root, dataset_name, model_name):
    dataset_name = _canonical_dataset_name(dataset_name)
    ckpt_path = Path(pretrained_models_root) / dataset_name / 'Pretrain' / model_name / 'best_model.pth'
    if ckpt_path.exists():
        return ckpt_path
    fallback = Path('/data') / dataset_name / 'Pretrain' / model_name / 'best_model.pth'
    if fallback.exists():
        return fallback
    return ckpt_path


def _clean_state_dict_keys(state):
    new_state_dict = {}
    for k, v in state.items():
        if k.startswith('1.'):
            name = k[2:]
        elif k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def _head_state_from_checkpoint(ckpt):
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    state = _clean_state_dict_keys(state)
    if 'head.weight' in state:
        state = {k[len('head.'):]: v for k, v in state.items() if k.startswith('head.')}
    return state


def _is_dataset_special_model(model_name):
    name = model_name.lower()
    return (
        name == 'vmamba_small'
        or name.startswith('dinov3')
        or name.startswith('dinov2')
        or name.startswith('mae')
    )


def _build_vmamba_classifier(num_classes, pretrained=True, pretrained_models_root='./pretrained_models'):
    local_path = os.path.join(pretrained_models_root, 'vmamba_small')
    if not os.path.exists(local_path):
        local_path = '/home/lzr/learn/data/Vmamba_small'
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"VMamba local checkpoint not found: {local_path}")

    sys.path.insert(0, local_path)
    try:
        from configuration_vmamba import VMambaConfig
        from modeling_vmamba import VMambaForImageClassification

        config = VMambaConfig.from_pretrained(local_path)
        config.num_classes = num_classes
        if pretrained:
            hf_model = VMambaForImageClassification.from_pretrained(
                local_path, config=config, ignore_mismatched_sizes=True
            )
        else:
            hf_model = VMambaForImageClassification(config)
    finally:
        sys.path.pop(0)

    class VMambaWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(pixel_values=x).logits

    return VMambaWrapper(hf_model)


def _load_vmamba_model_for_dataset(model_name, dataset_name='CUB', pretrained_models_root='./pretrained_models'):
    model = _build_vmamba_classifier(
        _num_classes_for_dataset(dataset_name), pretrained=True,
        pretrained_models_root=pretrained_models_root
    )
    ckpt_path = _dataset_checkpoint_path(pretrained_models_root, dataset_name, model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = _clean_state_dict_keys(ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    accuracy = ckpt.get('accuracy', None) if isinstance(ckpt, dict) else None
    print(f"=> Loaded {model_name} from {ckpt_path}")
    if accuracy is not None:
        print(f"  Validation accuracy: {accuracy*100:.2f}%")
    return model


def _load_selfsup_model_for_dataset(model_name, dataset_name='CUB', pretrained_models_root='./pretrained_models'):
    from transformers import AutoConfig, AutoModel

    name = model_name.lower()
    model_id_map = {
        'dinov3':        'facebook/dinov3-small',
        'dinov3-vits16': 'facebook/dinov3-small',
        'dinov2-vits14': 'facebook/dinov2-small',
        'mae-base':      'facebook/vit-mae-base',
    }
    model_id = model_id_map.get(name)
    if model_id is None:
        raise ValueError(f"Unknown self-supervised model: {model_name}")

    local_name = model_id.split('/')[-1]
    model_path = os.path.join(pretrained_models_root, local_name)
    if not os.path.exists(model_path):
        model_path = model_id

    config = AutoConfig.from_pretrained(model_path)
    backbone = AutoModel.from_pretrained(model_path, config=config)
    model = DINOv3Classifier(backbone, num_classes=_num_classes_for_dataset(dataset_name))

    head_model_name = 'dinov3-vits16' if name == 'dinov3' else model_name
    ckpt_path = _dataset_checkpoint_path(pretrained_models_root, dataset_name, head_model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.head.load_state_dict(_head_state_from_checkpoint(ckpt))
    accuracy = ckpt.get('accuracy', None) if isinstance(ckpt, dict) else None
    print(f"=> Loaded {model_name} CUB head from {ckpt_path}")
    if accuracy is not None:
        print(f"  Validation accuracy: {accuracy*100:.2f}%")

    preprocess = PreprocessingModel(224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return nn.Sequential(preprocess, model)


def _load_dataset_special_model(model_name, dataset_name='CUB', pretrained_models_root='./pretrained_models'):
    if model_name.lower() == 'vmamba_small':
        return _load_vmamba_model_for_dataset(model_name, dataset_name, pretrained_models_root)
    return _load_selfsup_model_for_dataset(model_name, dataset_name, pretrained_models_root)


def load_single_model(model_name, dataset_name='ImageNet', pretrained_models_root='./pretrained_models'):
    """给main_mo/main_sparse等直接调用的单模型加载接口。"""
    if dataset_name.lower() != 'imagenet' and _is_dataset_special_model(model_name):
        return _load_dataset_special_model(model_name, dataset_name, pretrained_models_root)

    # DINOv3 models: try local cache, fallback to HuggingFace hub
    if model_name.lower().startswith('dinov3'):
        model_id_map = {
            'dinov3':        'facebook/dinov3-small',
            'dinov3-vits16': 'facebook/dinov3-small',
            'dinov3-vitb16': 'facebook/dinov3-base',
            'dinov3-vitl16': 'facebook/dinov3-large',
        }
        model_id = model_id_map.get(model_name.lower(), 'facebook/dinov3-small')
        local_name = model_id.split('/')[-1]
        dinov3_path = os.path.join(pretrained_models_root, local_name)
        if not os.path.exists(dinov3_path):
            dinov3_path = model_id  # auto-download from HuggingFace
        return _load_dinov3_model(dinov3_path)

    if dataset_name.lower() == 'imagenet':
        model = _load_pretrained_model_generic(model_name)
        return wrap_model(model)

    num_classes = _num_classes_for_dataset(dataset_name)

    ckpt_path = _dataset_checkpoint_path(pretrained_models_root, dataset_name, model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    print(f"=> Loading {model_name} from {ckpt_path}")
    model = get_model_architecture(model_name, num_classes)

    ckpt = torch.load(ckpt_path, map_location='cuda')
    state = ckpt.get('model_state_dict', ckpt)

    new_state_dict = _clean_state_dict_keys(state)
    
    model.load_state_dict(new_state_dict)

    # Get accuracy if available
    accuracy = ckpt.get('accuracy', None)
    
    print(f"Loaded pretrained model: {model_name} on {dataset_name}")
    if accuracy is not None:
        print(f"  Validation accuracy: {accuracy*100:.2f}%")
    print(f"  Model path: {ckpt_path}")

    return model

def load_pretrained_model(model_list, dataset_name='ImageNet', pretrained_models_root='./pretrained_models'):
    """
    支持按数据集加载模型：
    - ImageNet: 使用ImageNet预训练权重（torchvision/timm）
    - 其他(CUB/AIR/STCAR): 从pretrained_models/{dataset}/Pretrain/{model}/best_model.pth 加载
    """
    for model_name in model_list:
        if dataset_name.lower() == 'imagenet':
            model = _load_pretrained_model_generic(model_name, pretrained_models_root)
            yield model_name, wrap_model(model)
        else:
            if _is_dataset_special_model(model_name):
                yield model_name, _load_dataset_special_model(
                    model_name, dataset_name, pretrained_models_root
                )
                continue

            num_classes = _num_classes_for_dataset(dataset_name)

            ckpt_path = _dataset_checkpoint_path(pretrained_models_root, dataset_name, model_name)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

            print(f"=> Loading {model_name} from {ckpt_path}")
            model = get_model_architecture(model_name, num_classes)

            ckpt = torch.load(ckpt_path, map_location='cuda')
            state = ckpt.get('model_state_dict', ckpt)

            new_state_dict = _clean_state_dict_keys(state)
            
            model.load_state_dict(new_state_dict)

            # Get accuracy if available
            accuracy = ckpt.get('accuracy', None)
            
            print(f"Loaded pretrained model: {model_name} on {dataset_name}")
            if accuracy is not None:
                print(f"  Validation accuracy: {accuracy*100:.2f}%")
            print(f"  Model path: {ckpt_path}")
            yield model_name, model


class DINOv3Classifier(nn.Module):
    def __init__(self, backbone, num_classes=1000):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        # DINOv3: pooler_output;  MAE/ViT: last_hidden_state[:, 0]
        pooled = getattr(out, 'pooler_output', None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        return self.head(pooled)


def _load_dinov3_model(weights_path):
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(weights_path)
    backbone = AutoModel.from_pretrained(weights_path, config=config)
    model = DINOv3Classifier(backbone)
    # Load 5ep pre-trained head (99.9% acc on our data)
    head_path = './pretrained_models/dinov3-vits16-5ep-head.pth'
    if os.path.exists(head_path):
        model.head.load_state_dict(torch.load(head_path, map_location='cpu'))
        print(f"=> Loaded 5ep head (99.9% acc)")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess = PreprocessingModel(224, mean, std)
    return nn.Sequential(preprocess, model)


def _load_dinov3_model_raw(model_name, pretrained_models_root='./pretrained_models'):
    """
    Load DINOv3 backbone + classification head WITHOUT preprocessing.
    (wrap_model() adds preprocessing later.)
    Uses HuggingFace model hub, with optional local cache in pretrained_models/.
    """
    from transformers import AutoConfig, AutoModel

    model_id_map = {
        'dinov3':        'facebook/dinov3-small',
        'dinov3-vits16': 'facebook/dinov3-small',
        'dinov3-vitb16': 'facebook/dinov3-base',
        'dinov3-vitl16': 'facebook/dinov3-large',
    }
    model_id = model_id_map.get(model_name.lower(), 'facebook/dinov3-small')
    local_name = model_id.split('/')[-1]
    model_path = os.path.join(pretrained_models_root, local_name)
    if not os.path.exists(model_path):
        model_path = model_id  # auto-download from HuggingFace hub

    print(f"=> Loading {model_name} from DINOv3 pretrained ({model_path})")
    config = AutoConfig.from_pretrained(model_path)
    backbone = AutoModel.from_pretrained(model_path, config=config)
    model = DINOv3Classifier(backbone)
    head_path = os.path.join(pretrained_models_root, 'dinov3-vits16-5ep-head.pth')
    if os.path.exists(head_path):
        model.head.load_state_dict(torch.load(head_path, map_location='cpu'))
        print(f"=> Loaded 5ep head (99.9% acc)")
    return model


def _load_selfsup_model_raw(model_name, pretrained_models_root='./pretrained_models'):
    """
    Load self-supervised models (DINOv2, MAE) with ImageNet classifiers.
    Uses HuggingFace model hub with local cache fallback.
    """
    from transformers import AutoConfig, AutoModel

    model_id_map = {
        'dinov2-vits14': 'facebook/dinov2-small',
        'mae-base':      'facebook/vit-mae-base',
    }
    model_id = model_id_map.get(model_name.lower())
    if model_id is None:
        raise ValueError(f"Unknown self-supervised model: {model_name}")

    local_name = model_id.split('/')[-1]
    model_path = os.path.join(pretrained_models_root, local_name)
    if not os.path.exists(model_path):
        model_path = model_id  # auto-download from HuggingFace hub

    # Classification head paths
    head_map = {
        'dinov2-vits14': os.path.join(pretrained_models_root, 'dinov2-vits14-1k-head.pth'),
        'mae-base':      os.path.join(pretrained_models_root, 'mae-base-1k-head.pth'),
    }

    config = AutoConfig.from_pretrained(model_path)
    backbone = AutoModel.from_pretrained(model_path, config=config)
    model = DINOv3Classifier(backbone)

    head_path = head_map.get(model_name.lower())
    if head_path and os.path.exists(head_path):
        model.head.load_state_dict(torch.load(head_path, map_location='cpu'))
        print(f"=> Loaded trained head from {head_path}")
    return model


def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    model_name = model.__class__.__name__
    Resize = 224
    # import ipdb; ipdb.set_trace()
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        if 'Inc' in model_name:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224

    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model)


def save_images(output_dir, adversaries, filenames):
    adversaries_np = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255)
    adversaries_np = np.round(adversaries_np).astype(np.uint8)
    for i, filename in enumerate(filenames):
        # 2. 剥离任何可能导致有损压缩的后缀名 (如 .jpg, .jpeg)
        stem = os.path.splitext(filename)[0]
        safe_filename = f"{stem}.png"
        
        # 3. 强制使用 PNG 格式无损落盘
        Image.fromarray(adversaries_np[i]).save(os.path.join(output_dir, safe_filename), format='PNG')

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = resize
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # Use F.interpolate instead of transforms.Resize for CUDA compatibility
        x = torch.nn.functional.interpolate(
            x, size=(self.resize, self.resize),
            mode='bilinear', align_corners=False
        )
        return (x - self.mean) / self.std


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, dataset='ImageNet', input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.targeted = targeted
        self.target_class = target_class
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        self.transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.CenterCrop((img_height, img_width)),
                transforms.ToTensor(),
            ])
        self.dataset = dataset
        self.eval = eval

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        if self.eval:
            filepath = os.path.join(self.data_dir, os.path.splitext(filename)[0] + '.png')
        else:
            filepath = os.path.join(self.data_dir, filename)
        if self.dataset.lower() == 'imagenet':
            image = Image.open(filepath)
            image = image.resize((img_height, img_width)).convert('RGB')
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            image = np.array(image).astype(np.float32)/255
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image = Image.open(filepath).convert('RGB')
            if self.dataset == 'AIR':
                width, height = image.size
                image = image.crop((0, 0, width, height - 20))
            image = self.transform(image)

        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            if self.target_class:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], self.target_class] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=True, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
