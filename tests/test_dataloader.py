from app.ai.datasets.dataloader import create_dataloader


def main():

    loader = create_dataloader()

    features, labels = next(iter(loader))

    print("--------------------------------")
    print("Batch Loaded Successfully")
    print("--------------------------------")
    print("Feature Shape :", features.shape)
    print("Labels Shape  :", labels.shape)
    print("Feature Device:", features.device)
    print("--------------------------------")


if __name__ == "__main__":
    main()