import torch
import torch.nn as nn

# ----------------------------------------------------
# CNN MODEL
# ----------------------------------------------------

class DeepfakeCNN(nn.Module):

    def __init__(self):

        super(DeepfakeCNN, self).__init__()

        # ---------------------------------------------
        # CONVOLUTION BLOCK 1
        # ---------------------------------------------

        self.conv_block1 = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(0.2)
        )

        # ---------------------------------------------
        # CONVOLUTION BLOCK 2
        # ---------------------------------------------

        self.conv_block2 = nn.Sequential(

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(0.2)
        )

        # ---------------------------------------------
        # CONVOLUTION BLOCK 3
        # ---------------------------------------------

        self.conv_block3 = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(0.3)
        )

        # ---------------------------------------------
        # FULLY CONNECTED LAYERS
        # ---------------------------------------------

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
             64 * 16 * 12,
             128
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(
            128,
            2
        )

    # ----------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------

    def forward(self, x):

        x = self.conv_block1(x)

        x = self.conv_block2(x)

        x = self.conv_block3(x)

        x = self.flatten(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x