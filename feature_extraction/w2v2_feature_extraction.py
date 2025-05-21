from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import numpy as np

class Wav2Vec2FeatureExtraction:
    def __init__(self, feature_extractor, model, device):
        self.feature_extractor = feature_extractor
        self.model = model
        self.device = device

    def load_audio_file(self, path_to_audio_file = './', sample_rate=16000):

        """
        Load the audio file (or generate dummy audio).

        Args:
        path_to_audio_file (str/Tensor): Path to the audio file (or) waveform of the audio.
        sample_rate (int): The sample rate of the audio file.

        Returns:
        waveform (Tensor): The waveform tensor.
        sample_rate (int): The sample rate of the audio file.
        y (ndarray): The audio time series.
        duration (float): Duration of the audio.
        """
        try:
          waveform, sample_rate = torchaudio.load(path_to_audio_file)
        except:
          waveform = path_to_audio_file

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        waveform = waveform.to(self.device)
        y = waveform.squeeze().cpu().numpy()
        duration = len(y) / sample_rate
        return waveform, sample_rate, y, duration

    def global_feature_extraction(self, waveform, sample_rate):

        """
        Extract features of audio signal using the pretrained model.

        Args:
        waveform (Tensor): The waveform tensor.
        sample_rate (int): The sample rate of the audio file.

        Returns:
        features_np (ndarray): The extracted features as a numpy array.
        """

        inputs = self.feature_extractor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = inputs['input_values']
        with torch.no_grad():
            features = self.model(inputs.to(self.device)).hidden_states

        features_np = np.array([hs.cpu().numpy() for hs in features])
        features_np = features_np[0].mean(1)
        return features_np
