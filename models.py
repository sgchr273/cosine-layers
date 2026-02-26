from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Large_Weights
)

# ---- Your custom ResNet18 and DenseNet100 ----

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear  = nn.Linear(512*block.expansion, num_classes)
        self.linear1 = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4).view(out4.size(0), -1)
        logits_a  = self.linear(out5)
        logits_b  = self.linear1(out5)
        return logits_a, logits_b, out5, [out1, out2, out3, out4]

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)


class _DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, inter, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth_rate, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_ch, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(_DenseLayer(ch, growth_rate, bn_size, drop_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_ch = ch
    def forward(self, x):
        return self.block(x)

class _Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        x = self.conv(F.relu(self.norm(x), inplace=True))
        return self.pool(x)

class DenseNet100(nn.Module):
    def __init__(self, num_classes=10, growth_rate=12, block_config=(16,16,16),
                 compression=0.5, bn_size=4, drop_rate=0.0):
        super().__init__()
        init_ch = 2 * growth_rate
        self.conv0 = nn.Conv2d(3, init_ch, 3, 1, 1, bias=False)
        self.block1 = _DenseBlock(block_config[0], init_ch, growth_rate, bn_size, drop_rate)
        ch = int(self.block1.out_ch * compression)
        self.trans1 = _Transition(self.block1.out_ch, ch)

        self.block2 = _DenseBlock(block_config[1], ch, growth_rate, bn_size, drop_rate)
        ch = int(self.block2.out_ch * compression)
        self.trans2 = _Transition(self.block2.out_ch, ch)

        self.block3 = _DenseBlock(block_config[2], ch, growth_rate, bn_size, drop_rate)
        ch = self.block3.out_ch

        self.norm_final = nn.BatchNorm2d(ch)
        self.classifier = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = F.relu(self.norm_final(self.block3(x)), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.classifier(x)

# ---- Utility classes & functions ----

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelSpec:
    arch: str
    num_classes: int
    ckpt: Optional[str] = None

def set_all_seeds(seed: int = 0):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

def _maybe_strip_prefix(sd, prefixes=("module.",)):
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def load_ckpt(m: nn.Module, ckpt_path: str, arch: str, device: str = "cpu") -> nn.Module:
    """
    Load weights into `m` from a checkpoint.
    - If arch == 'resnet18' and checkpoint has 'ema', load EMA weights (strict=True).
    - Otherwise load 'model' (or raw state_dict) with strict=False.
    Moves the model to `device`, sets eval(), and returns it.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if arch.lower() == "resnet18" and isinstance(ckpt, dict) and ("ema" in ckpt):
        state = _maybe_strip_prefix(ckpt["ema"])
        m.load_state_dict(state, strict=True)
    else:
        state = _maybe_strip_prefix(ckpt.get("model", ckpt))
        m.load_state_dict(state, strict=False)

    m.to(device).eval()
    return m

def build_model(spec: ModelSpec) -> nn.Module:
    arch = spec.arch.lower()
    nc = spec.num_classes

    # arches that should ignore external checkpoints but still use torchvision pretrained weights
    ignore_ckpt_arches = {"resnet50", "mobilenet"}

    def maybe_load_ckpt(m):
        if spec.ckpt and arch not in ignore_ckpt_arches:
            load_ckpt(m, spec.ckpt, arch=arch, device=DEVICE)
        elif spec.ckpt and arch in ignore_ckpt_arches:
            print(f"[info] Ignoring ckpt '{spec.ckpt}' for arch '{arch}' (using torchvision pretrained only).")
        return m

    if arch == "densenet100":
        m = DenseNet100(num_classes=nc)
        return maybe_load_ckpt(m).to(DEVICE)

    elif arch == "resnet18":
        m = ResNet18(num_classes=nc)
        return maybe_load_ckpt(m).to(DEVICE)

    elif arch == "resnet50":
        # keep torchvision pretrained weights
        m = tvm.resnet50(weights=ResNet50_Weights.DEFAULT)
        # do NOT load spec.ckpt (blocked above)
        return m.to(DEVICE)

    elif arch == "mobilenet":
        # keep torchvision pretrained weights
        m = tvm.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        # do NOT load spec.ckpt (blocked above)
        return m.to(DEVICE)

    else:
        raise ValueError(f"Unknown architecture: {spec.arch}")




def _gap(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

@torch.no_grad()
def forward_adapt(model: nn.Module, x: torch.Tensor, arch: str
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
    a = arch.lower()

    # -------- DenseNet100 (your CIFAR-style) --------
    if a.startswith("densenet"):
        x0  = model.conv0(x)

        b1  = model.block1(x0)
        t1  = model.trans1(b1)

        out2 = model.block2(t1)              # "out2"
        t2   = model.trans2(out2)

        out3 = model.block3(t2)              # "out3"

        out4 = F.relu(model.norm_final(out3), inplace=True)  # "out4"

        penult = _gap(out4)
        logits = model.classifier(penult)
        logits_aux = logits
        extra = {1: out2, 2: out3, 3: out4, 4:penult}
        return logits, logits_aux, penult, extra

    # -------- ResNet-like (your implementation) --------
    if a.startswith("resnet18"):
        m = model
        y = m.conv1(x); y = m.bn1(y); y = F.relu(y)   # no m.relu attr
        # no maxpool in your CIFAR-style ResNet
        y = m.layer1(y)
        out2 = m.layer2(y)            # "out2"
        out3 = m.layer3(out2)         # "out3"
        out4 = m.layer4(out3)         # "out4"

        penult = _gap(out4)
        logits = m.linear(penult)       # no m.fc attr
        logits_aux = m.linear1(penult)  # or set = logits if desired
        extra = {1: out2, 2: out3, 3: out4}
        return logits, logits_aux, penult, extra

    if a.startswith("resnet50"):
        m = model
        y = m.conv1(x); y = m.bn1(y); y = m.relu(y); y = m.maxpool(y)
        y = m.layer1(y)
        out2 = m.layer2(y)            # "out2"
        out3 = m.layer3(out2)         # "out3"
        out4 = m.layer4(out3)         # "out4"

        penult = _gap(out4)           # m.avgpool(out4).flatten(1)
        logits = m.fc(penult)         # torchvision head
        logits_aux = logits
        extra = {1: out2, 2: out3, 3: out4, 4: penult}
        # extra = {1: out2, 2: out3, 3: out4}
        return logits, logits_aux, penult, extra

    # -------- MobileNetV3-Large (torchvision) --------
    if a.startswith("mobile") or a.startswith("mobilenet"):
        m = model
        y = x
        downs = []
        prev_hw = y.shape[-2:]

        # walk the feature extractor; record tensors when spatial size shrinks
        for layer in m.features:
            y = layer(y)
            hw = y.shape[-2:]
            if hw != prev_hw:
                downs.append(y)
                prev_hw = hw

        # y is the final feature map after m.features
        out_last = y
        # choose the deepest three stages: (last two downsamples) + (final map)
        if len(downs) >= 2:
            out2, out3 = downs[-2], downs[-1]
        elif len(downs) == 1:
            out2, out3 = downs[0], downs[0]
        else:
            out2 = out3 = out_last
        out4 = out_last

        penult = _gap(out_last)       # equivalent to m.avgpool(out_last).flatten(1)
        logits = m.classifier(penult) # torchvision head (Sequential)
        logits_aux = logits
        extra = {1: out2, 2: out3, 3: out4, 4:penult}
        return logits, logits_aux, penult, extra

    raise ValueError(f"Unknown architecture: {arch}")


def _get_penultimate(model: nn.Module, x: torch.Tensor, arch: str) -> torch.Tensor:
    if arch.startswith("resnet"):
        _, _, penult, _ = model(x)
        return penult
    if arch == "densenet100":
        feat = model.block3(model.trans2(model.block2(model.trans1(model.block1(model.conv0(x))))))
        pooled = F.adaptive_avg_pool2d(F.relu(model.norm_final(feat)), (1, 1)).flatten(1)
        return pooled
    if arch == "vit":
        grabbed = {}
        def hook_fn(_, __, output): grabbed["feat"] = output.flatten(1).detach()
        h = model.encoder.ln.register_forward_hook(hook_fn)
        _ = model(x); h.remove()
        return grabbed["feat"].to(x.device)
    if arch == "swin":
        grabbed = {}
        def hook_fn(_, __, output): grabbed["feat"] = output.flatten(1).detach()
        h = model.norm.register_forward_hook(hook_fn)
        _ = model(x); h.remove()
        return grabbed["feat"].to(x.device)
    raise RuntimeError(f"Unsupported arch for penultimate: {arch}")

# def _get_resnet_intermediates(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
#     _, _, _, outs = model(x)
#     return outs

@torch.no_grad()
def _get_resnet_intermediates(m: nn.Module, x: torch.Tensor) -> Dict[int, torch.Tensor]:
    # torchvision.resnet50 forward broken into stages
    y = m.conv1(x); y = m.bn1(y); y = m.relu(y); y = m.maxpool(y)
    y = m.layer1(y)          # conv2_x in original paper
    out2 = m.layer2(y)       # conv3_x
    out3 = m.layer3(out2)    # conv4_x
    out4 = m.layer4(out3)    # conv5_x
    return {1: out2, 2: out3, 3: out4}


def get_classifier_params(model: nn.Module):
    import numpy as np
    m = getattr(model, "module", model)
    for attr in ("classifier", "linear", "linear1", "fc", "head"):
        layer = getattr(m, attr, None)
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(W.shape[0])
            return W, b
    raise AttributeError("No classifier layer found.")
