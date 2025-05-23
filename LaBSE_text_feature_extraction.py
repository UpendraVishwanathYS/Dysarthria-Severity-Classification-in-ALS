import importlib
import os
import torch
import pandas as pd
import sys

def execute_LaBSE_text_feature_extraction(csv_path, transcription_type='HSR', info_type='with image',
                                    method='LaBSE', save=False, save_dir='./text_features'):

    sys.path.append(os.path.abspath('.'))

    data = pd.read_csv(csv_path)

    if info_type != 'all' and transcription_type == 'HSR':
        data = data[data['Type'] == info_type]

    transcripts_dict = data.groupby('File_name')['Transcript'].apply(list).to_dict()
    sentence_pairs = list(transcripts_dict.values())

    module = importlib.import_module('text_feature_extraction.labse_embeddings')
    extractor_class = getattr(module, 'LaBSE')

    extractor = extractor_class()

    _, avg_embeddings = extractor.encode_pairs(sentence_pairs)


    if save:
        os.makedirs(save_dir, exist_ok=True)
        for fname, emb in zip(transcripts_dict.keys(), avg_embeddings):
            out_path = os.path.join(save_dir, f"{fname}_{transcription_type}_{info_type}_{method}_features.pt")
            torch.save(emb, out_path)
            print(f"Saved: {out_path}")

    return avg_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text features using LaBSE or LASER")

    parser.add_argument('--csv_path',type=str,required=True,help='Path toCSV file containing transcription data')

    parser.add_argument( '--transcription_type',type=str,default='HSR',choices=['HSR','ASR'],help='Type of transcription to process (e.g., HSR)')

    parser.add_argument('--info_type',type=str,default='all',choices=['with image', 'without image', 'all'],help='Subset of transcripts to use: "with image", "without image", or "all"')

    parser.add_argument('--save',type=str,default='False',choices=['True', 'False'],help='Whether to save the extracted features')

    parser.add_argument('--save_dir',type=str,default='./text_features',help='Directory to save the features')

    args = parser.parse_args()

    avg_embeddings = execute_LaBSE_text_feature_extraction(
        csv_path=args.csv_path,
        transcription_type=args.transcription_type,
        info_type=args.info_type,
        save=args.save.lower() == 'true',
        save_dir=args.save_dir
    )
