import torch

from app.ai.models.cnn_model import DeepFakeCNN


def main():

    model = DeepFakeCNN()

    dummy_input = torch.randn(
        32,
        1,
        128,
        100
    )

    output = model(dummy_input)

    print("----------------------------")
    print("CNN Test Successful")
    print("----------------------------")
    print("Input Shape :", dummy_input.shape)
    print("Output Shape:", output.shape)
    print("----------------------------")


if __name__ == "__main__":
    main()