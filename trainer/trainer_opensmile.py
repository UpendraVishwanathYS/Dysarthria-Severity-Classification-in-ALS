from trainer.trainer import *
from models.dnn import *

class ALSModelTrainerOpenSMILE:
    def __init__(self, train_data_path, test_data_path,  opensmile_feature_path, classification_type, exp_name):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.classification_type = classification_type
        self.exp_name = exp_name
        self.opensmile_feature_path = opensmile_feature_path
        self.fold_metrics = []


    def three_class_problem(self, labels):
        label_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
        grouped_labels = np.array([label_mapping[label] for label in labels])
        return grouped_labels

    def two_class_problem(self, labels):
        label_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
        grouped_labels = np.array([label_mapping[label] for label in labels])
        return grouped_labels


    def normalization(self, feats, avg_trainfeat, std_trainfeat):
        return (feats - avg_trainfeat) / std_trainfeat

    def process_data(self, i):
        exp_name = f'{self.exp_name}_Combination{i}'

        opensmile_features_data = pd.read_csv(self.opensmile_feature_path)
        train_files = pd.read_excel(self.train_data_path, sheet_name=f'Combinations{i}')
        test_files = pd.read_excel(self.test_data_path, sheet_name=f'Combinations{i}')

        train_data = opensmile_features_data[train_files['Audio_file_name'].unique()]
        train_labels = [train_files.loc[train_files['Audio_file_name'] == x, 'Severity'].values[0] for x in train_data.columns]
        test_data = opensmile_features_data[test_files['Audio_file_name'].unique()]
        test_labels = [test_files.loc[test_files['Audio_file_name'] == x, 'Severity'].values[0] for x in test_data.columns]

        train_data = train_data.values.transpose()
        test_data = test_data.values.transpose()

        if self.classification_type == '3_class':
            train_labels = self.three_class_problem(train_labels)
            test_labels = self.three_class_problem(test_labels)
        if self.classification_type == '2_class':
            train_labels = self.two_class_problem(train_labels)
            test_labels = self.two_class_problem(test_labels)

        # Compute mean and std for normalization
        avg_trainfeat1 = np.mean(train_data, axis=0)
        std_trainfeat1 = np.std(train_data, axis=0)

        #normalized_train_data = self.normalization(train_data, avg_trainfeat1, std_trainfeat1)
        #normalized_test_data = self.normalization(test_data, avg_trainfeat1, std_trainfeat1)

        # Convert to tensors
        train_data_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
        test_data_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)

        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

        return train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor

    def train_model(self, train_loader, test_loader, train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor,exp_name):
        model = SimpleDNN_drop(train_data_tensor.shape[-1], num_classes=torch.unique(train_labels_tensor).shape[0])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainer = Trainer(model, criterion, optimizer, device, save_path=f'/content/drive/MyDrive/Detection of ALS/DNN/best_models/{exp_name}_best_model.pth', log_dir='/content/drive/MyDrive/Detection of ALS/DNN/logs', exp_name=f'{exp_name}')

        trainer.train(train_loader, test_loader, epochs=100)

        model = trainer.load_model(f'/content/drive/MyDrive/Detection of ALS/DNN/best_models/{exp_name}_best_model.pth')

        return model

    def evaluate_model(self, model, test_loader,device='cuda'):
        with torch.no_grad():
            model.eval()
            all_labels = []
            all_preds = []
            for inputs, labels in iter(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro')
        test_precision = precision_score(all_labels, all_preds, average='macro')
        test_recall = recall_score(all_labels, all_preds, average='macro')

        return test_acc, test_f1, test_precision, test_recall

    def run(self):
        for i in range(5):
            train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor = self.process_data(i)

            # Create Dataloaders
            train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
            test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
            batch_size = 32
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Train and evaluate model
            model = self.train_model(train_loader, test_loader, train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor, f'{self.exp_name}_Combination{i}')
            test_acc, test_f1, test_precision, test_recall = self.evaluate_model(model, test_loader)

            # Log and store metrics
            self.fold_metrics.append({
                'Fold': i,
                'Accuracy': test_acc,
                'F1 Score': test_f1,
                'Precision': test_precision,
                'Recall': test_recall
            })

            # Print fold metrics
            print(f'Combinations:{i}')
            print(f'Test Accuracy: {test_acc:.2f}, Test F1 Score: {test_f1:.2f}, Test Precision: {test_precision:.2f}, Test Recall: {test_recall:.2f}')

        # Calculate average and standard deviation
        metrics_df = pd.DataFrame(self.fold_metrics)
        self.print_average_metrics(metrics_df)

    def print_average_metrics(self, metrics_df):
        average_accuracy = np.mean(metrics_df['Accuracy'])
        std_accuracy = np.std(metrics_df['Accuracy'])
        average_f1 = np.mean(metrics_df['F1 Score'])
        std_f1 = np.std(metrics_df['F1 Score'])
        average_precision = np.mean(metrics_df['Precision'])
        std_precision = np.std(metrics_df['Precision'])
        average_recall = np.mean(metrics_df['Recall'])
        std_recall = np.std(metrics_df['Recall'])

        # Print the averages and standard deviations
        print("\nAverage and Standard Deviation across all folds:")
        print(f"Accuracy: {average_accuracy:.2f} ± {std_accuracy:.2f}")
        print(f"F1 Score: {average_f1:.2f} ± {std_f1:.2f}")
        print(f"Precision: {average_precision:.2f} ± {std_precision:.2f}")
        print(f"Recall: {average_recall:.2f} ± {std_recall:.2f}")

        # Append the average metrics to the DataFrame
        metrics_df.loc['Average ± Std'] = [None, f"{average_accuracy:.2f} ± {std_accuracy:.2f}", f"{average_f1:.2f} ± {std_f1:.2f}", f"{average_precision:.2f} ± {std_precision:.2f}", f"{average_recall:.2f} ± {std_recall:.2f}"]
        metrics_df.to_csv(f'/content/drive/MyDrive/Detection of ALS/DNN/results/{self.exp_name}.csv')

