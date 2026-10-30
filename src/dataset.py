import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# alphabetical order, same as the folder names
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def collect(root, split):
    root = Path(root)
    items = []
    for label, cls in enumerate(CLASSES):
        if split == "train":
            files = list((root / "train" / cls).glob("*"))
        else:
            prefix = "publictest_" if split == "val" else "privatetest_"
            files = [f for f in (root / "test" / cls).glob("*")
                     if f.name.lower().startswith(prefix)]
        items += [(f, label) for f in files
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return items


class FER2013(Dataset):
    """split: 'train' | 'val' (PublicTest) | 'test' (PrivateTest)"""

    def __init__(self, root, split, train=False):
        items = collect(root, split)
        self.images = np.stack(
            [np.array(Image.open(f).convert("L").resize((48, 48))) for f, _ in items]
        )
        self.labels = np.array([l for _, l in items], dtype=np.int64)
        print(f"{split}: {len(self.labels)} images")

        if train:
            self.tf = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.RandomCrop(48, padding=4),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ])
        else:
            self.tf = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(self.images[i]), int(self.labels[i])