from utils.required_libraries import *

class Trainer:
    def __init__(self, model, criterion, optimizer, device, save_path='best_model.pth', log_dir='runs',
                 exp_name='experiment_name', patience=5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.best_val_f1 = 0.0  # Track best F1-score
        self.patience = patience
        self.early_stop_counter = 0

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, exp_name))

    def train(self, train_loader, val_loader=None, epochs=10):
        self.model.to(self.device)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in iter(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

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
                val_f1 = self.evaluate(val_loader, epoch)

                # Early stopping check
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.save_model(self.save_path)
                    self.early_stop_counter = 0  # Reset counter if validation improves
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        print(f'Early stopping triggered after {epoch+1} epochs.')
                        break

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        all_labels = []
        all_preds = []
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in iter(val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'Validation Loss: {val_loss:.4f}, Validation F1 Score: {val_f1:.2f}')
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('F1/val', val_f1, epoch)

        return val_f1

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Best model (based on F1-score) saved to {path}')

    def load_model(self, path):
      self.model.load_state_dict(torch.load(path))
      model = self.model.to(self.device)
      print(f'Model loaded from {path}')
      return model
