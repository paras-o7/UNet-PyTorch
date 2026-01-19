import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

doubleConv = lambda in_channels, out_channels: nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
)

class UNet(nn.Module):
    def __init__(
        self,
        channels: list[int] = [64, 128, 256, 512, 1024],
        in_channels: int = 3,
        out_channels: int = 5,
    ) -> None:
        super().__init__()

        # self.down_layers: list[nn.Module] = [DoubleConv(in_channels, channels[0])]
        self.down_layers = nn.ModuleList()
        self.down_layers.append(doubleConv(in_channels, channels[0]))

        for i in range(len(channels) - 1): #4
            self.down_layers.append(
                nn.Sequential(
                    nn.MaxPool2d(2, stride=2),
                    doubleConv(channels[i], channels[i + 1])
                )
            )

        self.up_layers = nn.ModuleList()
        self.up_layers.append(nn.ConvTranspose2d(channels[-1], channels[-1], 2, stride=2))

        for i in reversed(range(2, len(channels))):
            self.up_layers.append(
                nn.Sequential(
                    doubleConv(channels[i] + channels[i-1], channels[i - 1]),
                    nn.ConvTranspose2d(channels[i - 1], channels[i - 1], 2, stride=2),
                )
            )

        self.up_layers.append(doubleConv(channels[1]+channels[0], channels[0]))

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skip = []
        for i in range(len(self.down_layers)):
            x = self.down_layers[i](x)
            skip.append(x)
        skip.pop()

        skip.reverse()
        for i in range(len(self.up_layers)-1):
            x = self.up_layers[i](x)
            skip_connection = TF.center_crop(skip[i], output_size=x.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
        x = self.up_layers[-1](x)
        A = self.final_conv(x)
        return A