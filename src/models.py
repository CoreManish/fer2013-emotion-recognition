import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. Baseline CNN ----------
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 64), block(64, 128), block(128, 256), block(256, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- 2. Squeeze-and-Excitation block ----------
class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=(2, 3))                # squeeze
        w = self.fc(w)[:, :, None, None]      # excite
        return x * w                          # scale


# ---------- 3. ResNet-18 (optionally with SE) ----------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SE(cout) if use_se else nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                    # SE before the skip addition
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=7, use_se=False):
        super().__init__()
        # 3x3 stride-1 stem and no max-pool, because inputs are only 48x48
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        cfg = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
               (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]
        self.layers = nn.Sequential(*[BasicBlock(i, o, s, use_se) for i, o, s in cfg])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


def get_model(name):
    if name == "baseline":
        return BaselineCNN()
    if name == "resnet18":
        return ResNet18(use_se=False)
    if name == "se_resnet18":
        return ResNet18(use_se=True)
    raise ValueError(name)