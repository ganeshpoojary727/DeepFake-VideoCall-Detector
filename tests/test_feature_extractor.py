from app.config.settings import settings

from app.ai.preprocessing.audio_preprocessor import AudioPreprocessor
from app.ai.preprocessing.feature_extractor import FeatureExtractor


def main():

    preprocessor = AudioPreprocessor()

    extractor = FeatureExtractor()

    sample_audio = (
        settings.DATASET_DIR
        / "LA"
        / "ASVspoof2019_LA_train"
        / "flac"
        / "LA_T_1000137.flac"
    )

    audio, sr = preprocessor.preprocess(sample_audio)

    feature = extractor.extract(audio)

    print("----------------------------")
    print("Feature Extraction Success")
    print("----------------------------")
    print("Tensor Shape :", feature.shape)
    print("Tensor Type  :", feature.dtype)
    print("Device       :", feature.device)
    print("----------------------------")


if __name__ == "__main__":
    main()