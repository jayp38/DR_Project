# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import AptosDataset
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau


OUT_DIR = "out"

# custom CNN model. feel free to modify and add more layers if needed
class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class AptosDataModule(L.LightningDataModule):
    def __init__(self, csv_file, root_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # augmentations and normalization
        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # only normalize for validation
        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = AptosDataset(self.csv_file, self.root_dir, transform=self.train_transform)
            val_dataset = AptosDataset(self.csv_file, self.root_dir, transform=self.val_transform)
            if os.path.exists('train_idx.npy') and os.path.exists('val_idx.npy'):
                train_idx = np.load('train_idx.npy')
                val_idx = np.load('val_idx.npy')
            else:
                indices = np.arange(len(train_dataset))
                np.random.seed(37)
                np.random.shuffle(indices)
                split = int(0.8 * len(train_dataset))
                train_idx, val_idx = indices[:split], indices[split:]
                np.save('train_idx.npy', train_idx)
                np.save('val_idx.npy', val_idx)
            self.train_ds = Subset(train_dataset, train_idx)
            self.val_ds = Subset(val_dataset, val_idx)
            
            # class weights
            labels = [train_dataset.df.iloc[idx, 1] for idx in train_idx]
            class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class AptosCNN(L.LightningModule):
    def __init__(self, lr=1e-3, num_classes=5):
        super().__init__()
        self.save_hyperparameters()
        self.model = CustomCNN(num_classes)
        self.criterion = None  # we set it on fit start
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []

    def on_fit_start(self):
        dm = self.trainer.datamodule
        class_weights = dm.class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, stage):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        preds = torch.argmax(pred, dim=1).detach().cpu()
        labels = y.detach().cpu()
        if stage == 'train':
            self.train_preds.append(preds)
            self.train_labels.append(labels)
        elif stage == 'val':
            self.val_preds.append(preds)
            self.val_labels.append(labels)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, 'val')

    def on_train_epoch_start(self):
        self.train_preds = []
        self.train_labels = []

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_preds).numpy()
        labels = torch.cat(self.train_labels).numpy()
        acc = (preds == labels).mean()
        kappa = cohen_kappa_score(labels, preds, weights='quadratic')
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        self.log('train_acc', acc)
        self.log('train_kappa', kappa)
        self.log('train_macro_prec', prec)
        self.log('train_macro_rec', rec)
        self.log('train_macro_f1', f1)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()
        acc = (preds == labels).mean()
        kappa = cohen_kappa_score(labels, preds, weights='quadratic')
        bal = balanced_accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        self.log('val_acc', acc)
        self.log('val_kappa', kappa)
        self.log('val_bal_acc', bal)
        self.log('val_macro_prec', prec)
        self.log('val_macro_rec', rec)
        self.log('val_macro_f1', f1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_kappa', # we want to maximize kappa score. feel free to change this to val_loss and min mode to minimize loss if you guys want
                'interval': 'epoch',
                'frequency': 1
            }
        }

def train_model(epochs=15, batch_size=64, lr=1e-3):
    dm = AptosDataModule('train.csv', 'preprocessed_train_images', batch_size=batch_size)
    model = AptosCNN(lr=lr)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_kappa',
        dirpath=OUT_DIR,
        filename='best_model',
        save_top_k=1,
        mode='max',
    )
    logger = CSVLogger(save_dir=OUT_DIR, name='train_log')
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )
    trainer.fit(model, dm)

    # save state_dict for compatibility
    best_path = checkpoint_callback.best_model_path
    if best_path:
        checkpoint = torch.load(best_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        torch.save(model_state_dict, os.path.join(OUT_DIR, "best_model.pth"))

    # compute confusion matrix and classification report on validation set using the best model
    val_dl = dm.val_dataloader()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_model = CustomCNN()
    loaded_model.load_state_dict(model_state_dict)
    loaded_model.to(device)
    loaded_model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            pred = loaded_model(xb)
            val_preds.extend(torch.argmax(pred, dim=1).cpu().numpy())
            val_labels.extend(yb.cpu().numpy())
    cm = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(5))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix")
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()
    report = classification_report(val_labels, val_preds, zero_division=0)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # extract history from logs
    metrics_path = f"{logger.log_dir}/metrics.csv"
    metrics = pd.read_csv(metrics_path)
    history = {}
    for col in metrics.columns:
        if col not in ['epoch', 'step']:
            history[col] = metrics[col].dropna().tolist()
    val_kappas = history.get('val_kappa', [])
    if val_kappas:
        best_kappa = max(val_kappas)
        best_epoch = val_kappas.index(best_kappa)
        history['best_epoch'] = best_epoch
        history['best_kappa'] = best_kappa
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history

def plot_graphs(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.get("train_acc", []), label="Train Acc")
    plt.plot(history.get("val_acc", []),   label="Val Acc")
    if "best_epoch" in history:
        plt.axvline(history["best_epoch"], color='k', linestyle='--', label=f"Best {history['best_epoch']}")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.get("train_loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []),   label="Val Loss")
    if "best_epoch" in history:
        plt.axvline(history["best_epoch"], color='k', linestyle='--')
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "train_history.png"), dpi=200)

if __name__ == "__main__":
    history = train_model()
    plot_graphs(history)