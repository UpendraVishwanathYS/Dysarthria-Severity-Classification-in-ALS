# Dysarthria-Severity-Classification-in-ALS

Comparison of Acoustic and Textual Features for Dysarthria Severity Classification in Amyotrophic Lateral Sclerosis

<img width="500" alt="image" src="https://github.com/user-attachments/assets/cdf7eedd-d6b3-4d70-892e-d7589f51f2bd" />

# Installing Dependencies
To install dependencies, create a conda or virtual environment with Python 3 and then run ```pip install -r requirements.txt```

# Speech Feature Extraction
Execution of the following command will extract speech embeddings and save them as .pt files for valid audio files in the specified path: ```python extract_speech_features.py --method "W2V2" --audio_path "/content/audio_data/*.wav" --save_flag True --save_dir "./saved_features"```
</br>The default values are specified below:
```
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True, help="Feature extraction method: 'OpenSMILE' or 'W2V2'") 
parser.add_argument('--audio_path', type=str, required=True, help="path to audio files")  # path/*.wav
parser.add_argument('--save_flag', type=str, default='False', help="Whether to save each feature vector")
parser.add_argument('--save_dir', type=str, default='./features', help="Directory to save feature files")
```
# Text Feature Extraction
Execution of the following command will extract speech embeddings and save them as .pt files for the text transcriptions:
(a) LaBSE: ```python LaBSE_extract_text_features.py --csv_path "hsr_transcripts.csv" --transcription_type HSR --info_type "with image" --save True --save_dir "./labse_features"```
</br>(b) LASER: ```python LASER_extract_text_features.py --csv_path "hsr_transcripts.csv" --transcription_type HSR --info_type "with image" --save True --save_dir "./laser_features"```
</br>The default values are specified below:
```
parser = argparse.ArgumentParser(description="Extract text features using LaBSE (or) LASER")
parser.add_argument('--csv_path',type=str,required=True,help='Path to CSV file containing transcription data')
parser.add_argument( '--transcription_type',type=str,default='HSR',choices=['HSR','ASR'],help='Type of transcriptions')
parser.add_argument('--info_type',type=str,default='all',choices=['with image', 'without image', 'all'],help='Subset of information to use for HSR: "with image", "without image", or "all"')
parser.add_argument('--save',type=str,default='False',choices=['True', 'False'],help='Whether to save the extracted features')
parser.add_argument('--save_dir',type=str,default='./text_features',help='Directory to save the features')
```
Input CSV format for text feature extraction:
| File_name     | Transcript                                   | Type          |  Severity     |
|---------------|----------------------------------------------|---------------|---------------|
| audio_001.wav | '{CLICK} মাছ  {PAUSE}  রেডি আছে' | with image    | 3 |
