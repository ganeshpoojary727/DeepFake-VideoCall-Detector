from pathlib import Path
from app.config.settings import settings
from app.ai.preprocessing.audio_preprocessor import AudioPreprocessor


def main():

    preprocessor = AudioPreprocessor()

    sample_audio = (
    settings.DATASET_DIR
    / "LA"
    / "ASVspoof2019_LA_train"
    / "flac"
    / "LA_T_1000137.flac"
)
    print(settings.DATASET_DIR)
    print(sample_audio)
    print(sample_audio.exists())
    audio, sr = preprocessor.preprocess(sample_audio)

    print("--------------------------------")
    print("Audio Loaded Successfully")
    print("--------------------------------")
    print("Sample Rate :", sr)
    print("Audio Shape :", audio.shape)
    print("--------------------------------")

   
if __name__ == "__main__":
    main()