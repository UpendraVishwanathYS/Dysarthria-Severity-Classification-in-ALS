import importlib
import os
import glob
import torch
import sys

def execute_feature_extraction(audio_file_paths, feature_extraction_method, save=False, save_dir='./speech_features'):

    sys.path.append(os.path.abspath('.'))

    if feature_extraction_method == 'OpenSMILE':
        module = importlib.import_module('speech_feature_extraction.opensmile_feature_extraction')
        extractor_class = getattr(module, 'OpenSmileFeatureExtractor')

    elif feature_extraction_method == 'W2V2':
        module = importlib.import_module('speech_feature_extraction.w2v2_feature_extraction')
        extractor_class = getattr(module, 'Wav2Vec2FeatureExtraction')

    else:
        raise ValueError("Unsupported feature extraction method. Use 'OpenSMILE' or 'W2V2'.")


    feature_extractor = extractor_class()

    os.makedirs(save_dir, exist_ok=True) if save else None

    # features = torch.cat([
    # torch.tensor(feature_extractor.extract_features(x)).unsqueeze(0)
    # for x in audio_file_paths], dim=0)
    
    features = []
    for path in audio_files:
        feat = torch.tensor(feature_extractor.extract_features(path)).unsqueeze(0)
        features.append(feat)
        if save:
            torch.save(feat, os.path.join(save_dir, f"{os.path.splitext(os.path.basename(path))[0]}_{feature_extraction_method}_features.pt"))
          
      
    return torch.cat(features,dim=0)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, help="Feature extraction method: 'OpenSMILE' or 'W2V2'")      # W2V2 or OpenSMILE
    parser.add_argument('--audio_path', type=str, required=True, help="path to audio files")  # path/*.wav
    parser.add_argument('--save_flag', type=str, default='False', help="Whether to save each feature vector")
    parser.add_argument('--save_dir', type=str, default='./features', help="Directory to save feature files")
    args = parser.parse_args()

    files = glob.glob(args.audio_path)
    print(f"Found {len(files)} files")

    all_features = execute_feature_extraction(files, args.method, args.save_flag.lower() == 'true', args.save_dir)
    print(f"Feature shape: {all_features.shape}")
