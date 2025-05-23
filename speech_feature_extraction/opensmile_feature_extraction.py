import opensmile
import librosa
import numpy as np

class OpenSmileFeatureExtractor:
    def __init__(self, config='emobase'):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[config],
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def load_audio(self, file_path, sr=16000):
        """Load an audio file and return the audio data and sample rate."""
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate

    def extract_features(self, file_path):
        """Extract OpenSMILE features from an audio file."""
        audio, sample_rate = self.load_audio(file_path)
        features = self.smile.process_signal(audio, sample_rate)
        return features
