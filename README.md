# Dysarthria-Severity-Classification-in-ALS

Comparison of Acoustic and Textual Features for Dysarthria Severity Classification in Amyotrophic Lateral Sclerosis

<img width="500" alt="image" src="https://github.com/user-attachments/assets/cdf7eedd-d6b3-4d70-892e-d7589f51f2bd" />

# Installing Dependencies
(1) To install dependencies, create a conda or virtual environment with Python 3 and then run ```pip install -r requirements.txt```
</br> (2) For LASER embedding based experiments: create a conda or virtual environment with Python 3 and then execute ```bash laser_installation.sh```.

# Speech Feature Extraction
Execution of the following command will extract speech embeddings and save them as .pt files for valid audio files in the specified path: ```python extract_speech_features.py --method "W2V2" --audio_path "/content/audio_data/*.wav" --save_flag True --save_dir "./saved_features"```
</br>The default values are specified below:
```
parser.add_argument('--method', type=str, required=True, help="Feature extraction method: 'OpenSMILE' or 'W2V2'") 
parser.add_argument('--audio_path', type=str, required=True, help="path to audio files")  # path/*.wav
parser.add_argument('--save_flag', type=str, default='False', help="Whether to save each feature vector")
parser.add_argument('--save_dir', type=str, default='./features', help="Directory to save feature files")
```
# Multilingual Sentence Embedding
Execution of the following command will extract speech embeddings and save them as .pt files for the text transcriptions:
(a) LaBSE: ```python LaBSE_extract_text_features.py --csv_path "hsr_transcripts.csv" --transcription_type HSR --info_type "with image" --save True --save_dir "./labse_features"```
</br>(b) LASER: ```python LASER_extract_text_features.py --csv_path "hsr_transcripts.csv" --transcription_type HSR --info_type "with image" --save True --save_dir "./laser_features"```
</br>The default values are specified below:
```
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


# Training Classification Model:
To train the Deep Neural Network (DNN) on speech and text embeddings simply run ```python train.py```. You can also add custom parameters in the command line: 
```
python train.py 
  --data_dir <DATA_DIR>
  --train_xlsx <TRAIN_XLSX>
  --test_xlsx <TEST_XLSX>
  --model_type <MODEL_TYPE>
  --classification_type <CLASSIFICATION_TYPE>
  --exp_name <EXP_NAME>
  --model_save_dir <MODEL_SAVE_DIR>
  --log_dir <LOG_DIR>
  --results_dir <RESULTS_DIR>
  --lr <LEARNING_RATE>
  --epochs <EPOCHS>
  --patience <PATIENCE>
```
The default values are specified below:
```
parser.add_argument("--data_dir", type=str, default="./data_feature_vectors", help="Directory containing the feature embeddings (.pt files)")
parser.add_argument("--train_xlsx", type=str, default="./train.xlsx", help="Path to the Excel file containing training folds information")
parser.add_argument("--test_xlsx", type=str, default="./test.xlsx", help="Path to the Excel file containing testing folds information")
parser.add_argument("--classification_model_type", type=str, required=True, help="Type of classification model")
parser.add_argument("--classification_type", type=str, default="3_class", help="Classification type: 2_class, 3_class, 5_class")
parser.add_argument("--exp_name", type=str, default="experiment", help="Name of the experiment.")
parser.add_argument("--model_save_dir", type=str, default="./saved_models", help="Directory to save trained models")
parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to store logs")
parser.add_argument("--results_dir", type=str, default="./results", help="Directory to store results as dataframes")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="No. of training epochs.")
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
```
