from utils.required_libraries import *

class ALSModelTrainer:
    def __init__(self, classification_type, exp_name, model_save_dir, log_dir, results_dir, model_fn, criterion_fn, optimizer_fn, lr, epochs, patience=5):
        self.classification_type = classification_type
        self.exp_name = exp_name
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        self.results_dir = results_dir
        self.model_fn = model_fn
        self.criterion_fn = criterion_fn
        self.optimizer_fn = optimizer_fn
        self.lr = lr
        self.fold_metrics = []
        self.patience = patience
        self.best_val_f1 = 0.0
        self.early_stop_counter = 0
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, exp_name))
        self.epochs = epochs

    def three_class_problem(self, labels):
        label_mapping = torch.tensor([0, 0, 1, 1, 2])
        labels = torch.as_tensor(labels, dtype=torch.long)
        return label_mapping[labels]

    def two_class_problem(self, labels):
        label_mapping = torch.tensor([0, 0, 1, 1, 1])
        labels = torch.as_tensor(labels, dtype=torch.long)
        return label_mapping[labels]

    def normalization(self, feats, avg_trainfeat, std_trainfeat):
        return (feats - avg_trainfeat) / std_trainfeat

    def process_data(self, train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor):
        if self.classification_type == '3_class':
            train_labels = self.three_class_problem(train_labels_tensor)
            test_labels = self.three_class_problem(test_labels_tensor)
        elif self.classification_type == '2_class':
            train_labels = self.two_class_problem(train_labels_tensor)
            test_labels = self.two_class_problem(test_labels_tensor)

        train_data = train_data_tensor.numpy()
        test_data = test_data_tensor.numpy()

        avg_trainfeat = np.mean(train_data, axis=0)
        std_trainfeat = np.std(train_data, axis=0)

        normalized_train_data = self.normalization(train_data, avg_trainfeat, std_trainfeat)
        normalized_test_data = self.normalization(test_data, avg_trainfeat, std_trainfeat)

        train_data_tensor = torch.tensor(normalized_train_data, dtype=torch.float32).unsqueeze(1)
        test_data_tensor = torch.tensor(normalized_test_data, dtype=torch.float32).unsqueeze(1)

        return train_data_tensor, test_data_tensor, train_labels, test_labels

    def train_model(self, train_loader, val_loader, model, criterion, optimizer, device, save_path, epochs=epochs):
        model.to(device)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in iter(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

            epoch_loss = running_loss / len(train_loader)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')

            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, F1 Score: {epoch_f1:.2f}')
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('F1/train', epoch_f1, epoch)

            if val_loader:
                val_f1 = self.evaluate_model(model, val_loader, device, epoch)
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.save_model(model, save_path)
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        print(f'Early stopping triggered after {epoch+1} epochs.')
                        break

        return self.load_model(model, save_path, device)

    def evaluate_model(self, model, data_loader, device='cuda', epoch=None):
        model.eval()
        all_labels = []
        all_preds = []
        loss_total = 0.0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        if epoch is not None:
            self.writer.add_scalar('Loss/val', loss_total / len(data_loader), epoch)
            self.writer.add_scalar('F1/val', f1_score(all_labels, all_preds, average='macro'), epoch)

        return f1_score(all_labels, all_preds, average='macro')

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
        print(f'Best model (based on F1-score) saved to {path}')

    def load_model(self, model, path, device):
        model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
        return model.to(device)

    def run(self, data_splits):
        for i, (train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor) in enumerate(sorted(data_splits.keys())):
            exp_name = f'{self.exp_name}_Combination{i}'
            train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor = self.process_data(
                train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor
            )

            train_loader = DataLoader(TensorDataset(train_data_tensor, train_labels_tensor), batch_size=32, shuffle=True)
            test_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), batch_size=32, shuffle=False)

            model = self.model_fn(input_shape=train_data_tensor.shape[-1], num_classes=torch.unique(train_labels_tensor).shape[0])
            criterion = self.criterion_fn()
            optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            save_path = f'{self.model_save_dir}/{exp_name}_best_model.pth'
            model = self.train_model(train_loader, test_loader, model, criterion, optimizer, device, save_path)
            metrics = self.evaluate_model(model, test_loader, device)

            self.fold_metrics.append({'Fold': i, 'Accuracy': accuracy_score(test_labels_tensor, torch.argmax(model(test_data_tensor.to(device)), dim=1).cpu()), **{
                'F1 Score': metrics,
                'Precision': precision_score(test_labels_tensor, torch.argmax(model(test_data_tensor.to(device)), dim=1).cpu(), average='macro'),
                'Recall': recall_score(test_labels_tensor, torch.argmax(model(test_data_tensor.to(device)), dim=1).cpu(), average='macro')
            }})

            print(f'Combinations:{i}')
            print(", ".join([f'{k}: {v:.2f}' for k, v in self.fold_metrics[-1].items() if k != 'Fold']))

        metrics_df = pd.DataFrame(self.fold_metrics)
        self.print_average_metrics(metrics_df)

    def print_average_metrics(self, metrics_df):
        summary = {}
        for metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
            mean = np.mean(metrics_df[metric])
            std = np.std(metrics_df[metric])
            summary[metric] = f"{mean:.2f} ± {std:.2f}"

        print("\nAverage and Standard Deviation across all folds:")
        for k, v in summary.items():
            print(f"{k}: {v}")

        metrics_df.loc['Average ± Std'] = [None] + list(summary.values())
        metrics_df.to_csv(f'{self.results_dir}/{self.exp_name}.csv', index=False)
