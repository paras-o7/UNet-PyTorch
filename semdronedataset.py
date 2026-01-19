import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import cv2


class SemanticDroneDataset(Dataset):
    def __init__(self, images_dir: str, transform=None) -> None:
        self.transform = transform
        self.images_dir = images_dir
        self.images_path = os.path.join(images_dir, "original")
        self.masks_path = os.path.join(images_dir, "mask")
        self.images = sorted(os.listdir(os.path.join(images_dir, "original")))
        self.masks = sorted(os.listdir(os.path.join(images_dir, "mask")))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img = torch.tensor(
            np.transpose(np.array(
                Image.open(os.path.join(self.images_path, self.images[idx])).convert(
                    "RGB"
                )
            ), axes=(2, 0, 1)),
            dtype=torch.float32
        )
        mask = torch.tensor(
            np.array(
                Image.open(os.path.join(self.masks_path, self.masks[idx])),
            ),
            dtype=torch.long
        )

        if self.transform:
            img = self.transform(img)
            mask.unsqueeze_(0)
            mask = self.transform(mask)
            mask.squeeze_(0)
        
        return img, mask, self.images[idx]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data = SemanticDroneDataset("archive/classes_dataset/classes_dataset/")
    loader = DataLoader(data)
    for i, j, name in loader:
        print(i.shape, j.shape)
        # Check if the pair of input image and the segmented image are the same.
        cv2.imshow("a", np.transpose(i.numpy()[0], axes=(1, 2, 0)).astype(np.uint8))
        cv2.imshow("b", j.numpy()[0].astype(np.uint8)*40)
        cv2.waitKey(0)
        cv2.destroyAllWindows()