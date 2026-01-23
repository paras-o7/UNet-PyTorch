import torch
from torch.utils.data import DataLoader, Subset
from unet import UNet
from semdronedataset import SemanticDroneDataset
import numpy as np
import cv2 as cv
from PIL import Image
import torchvision.transforms.functional as TF


class UNet_Pred:
    def __init__(
        self,
        weights_path: str,
        device: str,
        channels: list[int],
        in_channels: int,
        out_channels: int,
        palette: dict[int, tuple[int, int, int]],
        trained_w_parallel: bool,
    ) -> None:
        self.unet = UNet(
            channels=channels,
            in_channels=in_channels,
            out_channels=out_channels
        )

        self.device = device
        state_dict = torch.load(
            weights_path,
            map_location=torch.device(self.device)
        )
        if not trained_w_parallel:
            self.unet.load_state_dict(state_dict)
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove "module."
                new_state_dict[name] = v
            self.unet.load_state_dict(new_state_dict)
            
        self.palette = palette
        
        self.unet.eval()
    def predict_from_image_path(self, img_path: str) -> np.ndarray:
        img = torch.tensor(
            np.transpose(np.array(
                Image.open(img_path).convert("RGB"),
                dtype=np.float32
            ), axes=(2, 0, 1))
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

if __name__ == "__main__":
    DEVICE = "cpu"
    PIN_MEMORY = False
    PALETTE = {
        0: (155, 38, 182),
        1: (14, 135, 204),
        2: (124, 252, 0),
        3: (255, 20, 147),
        4: (169, 169, 169)
    }
    CHANNELS = [64, 128, 256, 512, 1024]
    TRAINED_W_PARALLEL = True
    IN_CHANNELS, OUT_CHANNELS = 3, 5
    DATASET_PATH = "./archive/classes_dataset/classes_dataset"
    PARAMS = "./model_save/unet_0_0271.pth"
    TEST_SET_IDX = list(range(300, 401))


    model = UNet_Pred(
        PARAMS,
        DEVICE,
        CHANNELS,
        IN_CHANNELS,
        OUT_CHANNELS,
        PALETTE,
        TRAINED_W_PARALLEL,
    )

    dataset = SemanticDroneDataset(DATASET_PATH)
    test = Subset(dataset, indices=TEST_SET_IDX)
    loader = DataLoader(test, shuffle=False, pin_memory=PIN_MEMORY)


    for batch_outs, batch_classes, name in loader:
        
        batch_outs = batch_outs.to(DEVICE)
        img = model.predict_from_image(batch_outs)

        cv.imshow(name[0], cv.cvtColor(img, cv.COLOR_RGB2BGR))

        cv.waitKey(0)
        cv.destroyAllWindows()
