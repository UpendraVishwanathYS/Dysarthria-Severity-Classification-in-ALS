import argparse
import os
from utils.required_libraries import *
from training.trainer import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using ALSModelTrainer with 5-fold validation.")

    parser.add_argument("--data_dir", type=str, default="./data_feature_vectors", help="Directory containing the feature embeddings (.pt files)")
    parser.add_argument("--train_xlsx", type=str, default="./folds/train.xlsx", help="Path to the Excel file containing training folds information")
    parser.add_argument("--test_xlsx", type=str, default="./folds/test.xlsx", help="Path to the Excel file containing testing folds information")
    parser.add_argument("--classification_model_type", type=str, required=True, help="Type of classification model")
    parser.add_argument("--classification_type", type=str, default="3_class", help="Classification type: 2_class, 3_class, 5_class")
    parser.add_argument("--exp_name", type=str, default="experiment", help="Name of the experiment.")
    parser.add_argument("--model_save_dir", type=str, default="./saved_models", help="Directory to save trained models")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to store logs")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to store results as dataframes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="No. of training epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load fold names
    folds = pd.ExcelFile(args.train_xlsx).sheet_names
    data_splits = {
        (fold,): load_fold_data(fold, args.data_dir, args.train_xlsx, args.test_xlsx)
        for fold in folds
    }

    model_fn = lambda input_shape, num_classes: load_model(args.classification_model_type, input_shape, num_classes)
    criterion_fn = nn.CrossEntropyLoss
    optimizer_fn = lambda params, lr=args.lr: optim.Adam(params, lr=lr)

    trainer = ALSModelTrainer(
        classification_type=args.classification_type,
        exp_name=args.exp_name,
        model_save_dir=args.model_save_dir,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        model_fn=model_fn,
        criterion_fn=criterion_fn,
        optimizer_fn=optimizer_fn,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience
    )

    trainer.run(data_splits)

if __name__ == "__main__":
    main()
