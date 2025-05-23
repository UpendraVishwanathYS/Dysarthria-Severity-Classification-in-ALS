from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import numpy as np
import torchaudio

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

class Wav2Vec2FeatureExtraction:
    def __init__(self, feature_extractor=feature_extractor, model=model, device=device):
        self.feature_extractor = feature_extractor
        self.model = model
        self.device = device

    def load_audio_file(self, path_to_audio_file, sample_rate=16000):
        try:
            waveform, original_sr = torchaudio.load(path_to_audio_file)
        except:
            waveform = path_to_audio_file
            original_sr = sample_rate

        if original_sr != 16000:
            waveform = torchaudio.functional.resample(waveform, original_sr, 16000)

        waveform = waveform.to(self.device)
        y = waveform.squeeze().cpu().numpy()
        duration = len(y) / sample_rate
        return waveform, sample_rate, y, duration

    def extract_features(self, path_to_audio_file):
        waveform, sample_rate, y, duration = self.load_audio_file(path_to_audio_file)

        inputs = self.feature_extractor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = inputs['input_values'].to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        last_hidden_state = outputs.last_hidden_state 
        mean_features = last_hidden_state.mean(dim=1).squeeze().cpu()
        return mean_features
