import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.1
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Dropout2d(p=dropout)

        )

    def forward(self, x):

        return self.block(x)


class CNN_V2(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(
                in_channels=1,
                out_channels=32,
                dropout=0.05
            ),

            ConvBlock(
                in_channels=32,
                out_channels=64,
                dropout=0.10
            ),

            ConvBlock(
                in_channels=64,
                out_channels=128,
                dropout=0.15
            ),

            ConvBlock(
                in_channels=128,
                out_channels=256,
                dropout=0.20
            )

        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(4, 4)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 4 * 4, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.40),

            nn.Linear(512, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),

            nn.Linear(128, num_classes)

        )

        self._initialize_weights()


    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="relu"
                )

            elif isinstance(
                m,
                (nn.BatchNorm2d, nn.BatchNorm1d)
            ):

                nn.init.ones_(m.weight)

                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="relu"
                )

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(
            self.classifier[-1].weight
        )


    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = self.classifier(x)

        return x
