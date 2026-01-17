import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os


class SemanticDroneDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        original_foldername: str,
        masks_foldername: str,
        transform=None,
    ) -> None:
        self.transform = transform
        self.images_dir = images_dir
        self.images_path = os.path.join(images_dir, original_foldername)
        self.masks_path = os.path.join(images_dir, masks_foldername)
        self.images = os.listdir(os.path.join(images_dir, original_foldername))
        self.masks = os.listdir(os.path.join(images_dir, masks_foldername))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.tensor(
            np.transpose(
                np.array(
                    Image.open(
                        os.path.join(self.images_path, self.images[idx])
                    ).convert("RGB")
                ),
                axes=(2, 0, 1),
            ),
            dtype=torch.float32,
        )
        mask = torch.tensor(
            np.array(
                Image.open(os.path.join(self.masks_path, self.masks[idx])),
            ),
            dtype=torch.long,
        )

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        # img = img.permute(2, 0, 1)
        # mask = mask.permute(2, 0, 1)
        return img, mask
