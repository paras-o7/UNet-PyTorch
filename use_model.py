import torch
from torch.utils.data import DataLoader, Subset
from unet import UNet
from semdronedataset import SemanticDroneDataset
import numpy as np
import cv2 as cv
from PIL import Image
import torchvision.transforms.functional as TF

DEVICE = "cuda"
PIN_MEMORY = False
PALETTE = {
    0: (155, 38, 182),
    1: (14, 135, 204),
    2: (124, 252, 0),
    3: (255, 20, 147),
    4: (169, 169, 169),
}


class UNet_Pred:
    def __init__(
        self,
        weights_path: str,
        device: str,
        channels: list[int],
        in_channels: int,
        out_channels: int,
        palette: dict[int, tuple[int, int, int]],
    ) -> None:
        self.unet = UNet(
            channels=channels, in_channels=in_channels, out_channels=out_channels
        )
        self.device = device
        self.unet.load_state_dict(
            torch.load(
                weights_path, weights_only=True, map_location=torch.device(self.device)
            )
        )
        self.unet = self.unet.to(self.device)
        self.palette = palette

        self.unet.eval()

    def predict_from_image_path(self, img_path: str) -> np.ndarray:
        img = torch.tensor(
            np.transpose(
                np.array(Image.open(img_path).convert("RGB"), dtype=np.float32),
                axes=(2, 0, 1),
            )
        ).to(self.device)

        img.unsqueeze_(0)

        return self.predict_from_image(img)

    def predict_from_image(self, img: torch.Tensor) -> np.ndarray:
        img /= 255
        pred = self.unet(img)
        pred = pred.argmax(dim=1)
        mask = pred[0].cpu().numpy().astype(np.uint8)

        segmentation = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for k, v in self.palette.items():
            segmentation[mask == k] = v

        return segmentation


model = UNet_Pred(
    "../unetBESTparams.pth", "cuda:0", [64, 128, 256, 512, 1024], 3, 5, PALETTE
)

dataset = SemanticDroneDataset("./archive/classes_dataset/classes_dataset")
testloader = Subset(dataset, indices=list(range(300, 400)))
loader = DataLoader(dataset, shuffle=False, pin_memory=PIN_MEMORY)

for batch_outs, batch_classes, name in loader:
    batch_outs = batch_outs.to(DEVICE)
    img = model.predict_from_image(batch_outs)

    cv.imshow(name[0], cv.cvtColor(img, cv.COLOR_RGB2BGR))

    cv.waitKey(0)
    cv.destroyAllWindows()
