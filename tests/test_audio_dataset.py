from app.ai.datasets.audio_dataset import AudioDataset

def main():
    dataset = AudioDataset()

    feature, label = dataset[0]

    print("Feature Shape :", feature.shape)
    print("Label :", label)

if __name__ == "__main__":
    main()