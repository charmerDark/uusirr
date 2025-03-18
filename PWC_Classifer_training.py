import logging
import os
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm_loggable.auto import tqdm
from torchvision import transforms
from PIL import Image
from models.new_model import PWCNet
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the extended model with classification layers
class PWCNetClassifier(nn.Module):
    def __init__(self):
        super(PWCNetClassifier, self).__init__()
        self.pwcnet = PWCNet(args=None, subnet = True)
        self.num_features_before_fc = 2 * 224 * 224  # 2 output channels, 224 x 224 flow maps
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features_before_fc, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x1, x2):
        flow_dict = self.pwcnet.forward({'input1': x1, 'input2': x2})
        flow = flow_dict['flow'][-1][0]
        flow_flat = flow.view(flow.size(0), -1)
        output = self.classifier(flow_flat)
        return output

def load_pretrained_pwcnet_weights(model, weights_path):
    """
    Load pretrained PWCNet weights with proper key mapping to handle structure differences.
    
    Args:
        model: The PWCNetClassifier model
        weights_path: Path to the pretrained weights file
    """
    # Load the pretrained weights
    pretrained_state_dict = torch.load(weights_path)
    if 'state_dict' in pretrained_state_dict:
        pretrained_state_dict = pretrained_state_dict['state_dict']
    
    # Create a new state dict with mapped keys
    mapped_state_dict = {}
    
    # Map the keys from pretrained weights to current model
    for k, v in pretrained_state_dict.items():
        # Handle the case where keys in pretrained weights have '_model' 
        # but current model doesn't have this in the path
        if '_model.' in k:
            new_key = k.replace('_model.', 'pwcnet.')
            mapped_state_dict[new_key] = v
        else:
            # For other keys, try to use them directly
            mapped_state_dict[k] = v
    
    # Load mapped weights into the model
    model_state_dict = model.state_dict()
    
    # Count how many keys were successfully mapped
    matched_keys = 0
    for k in model_state_dict.keys():
        if k in mapped_state_dict:
            model_state_dict[k] = mapped_state_dict[k]
            matched_keys += 1
    
    # Log how many weights were successfully loaded
    logger.info(f"Successfully loaded {matched_keys} weights out of {len(model_state_dict)} model parameters")
    
    # Load the updated state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

# Dataset
class UltrasoundFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.patient_ids = []
        label_mapping = {'benign': 0, 'malignant': 1}
        
        for label in ['benign', 'malignant']:
            class_dir = os.path.join(root_dir, label)
            for patient in os.listdir(class_dir):
                patient_dir = os.path.join(class_dir, patient)
                image_files = sorted(os.listdir(patient_dir))
                
                for i in range(len(image_files) - 1):
                    img_path1 = os.path.join(patient_dir, image_files[i])
                    img_path2 = os.path.join(patient_dir, image_files[i + 1])
                    self.samples.append((img_path1, img_path2, label_mapping[label]))
                    self.patient_ids.append(patient)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path1, img_path2, label = self.samples[idx]
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return (img1, img2), label

def patient_split(dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split dataset at the patient level into train, validation and test sets
    """
    # Get unique patients
    unique_patients = list(set(dataset.patient_ids))
    
    # Split patients
    train_patients, temp_patients = train_test_split(
        unique_patients, train_size=train_size, random_state=random_state
    )
    
    # Adjust val_size relative to the remaining data
    relative_val_size = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients, train_size=relative_val_size, random_state=random_state
    )
    
    # Create indices for each split
    train_indices = [i for i, patient in enumerate(dataset.patient_ids) if patient in train_patients]
    val_indices = [i for i, patient in enumerate(dataset.patient_ids) if patient in val_patients]
    test_indices = [i for i, patient in enumerate(dataset.patient_ids) if patient in test_patients]
    
    # Log the splits
    logger.info(f"Patient-level splits created:")
    logger.info(f"  Train: {len(train_patients)} patients, {len(train_indices)} samples")
    logger.info(f"  Validation: {len(val_patients)} patients, {len(val_indices)} samples")
    logger.info(f"  Test: {len(test_patients)} patients, {len(test_indices)} samples")
    
    return train_indices, val_indices, test_indices

def evaluate(model, data_loader, device):
    """
    Evaluate model performance on given data loader
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (img1, img2), labels in data_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return accuracy, precision, recall, f1

def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint saved: {filename}")

def plot_training_curves(metrics, save_path):
    """
    Plot training curves
    """
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['train_acc'], label='Train Accuracy')
    plt.plot(epochs, metrics['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['train_precision'], label='Train Precision')
    plt.plot(epochs, metrics['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    
    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['train_recall'], label='Train Recall')
    plt.plot(epochs, metrics['val_recall'], label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training curves saved to {save_path}")

# Training code in main function
def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging for each run
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'), 'a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # When training on cluster, ensure root_dir is in scratch for fast memory access
    root_dir = '/home/s2751455/uusirr/miccai_2022_buv_dataset/rawframes'
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #TODO: Using imageNet mean and std here, calculate for dataset and shift to it
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = UltrasoundFrameDataset(root_dir=root_dir, transform=transform)
    
    # Patient-level splits
    train_indices, val_indices, test_indices = patient_split(dataset)
    
    # Create data loaders
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)
    
    # Initialize model
    pwc_weights_path = "/home/s2751455/uusirr/saved_check_point/new_model/PWC_net_sintel/checkpoint_best.ckpt"
    model = PWCNetClassifier()

    model.to(device)
    model = load_pretrained_pwcnet_weights(model, pwc_weights_path)

    # Optimizer and scheduler
    optimizer = optim.Adam([
        {'params': model.pwcnet.parameters(), 'lr': 1e-5},  # Lower learning rate for pretrained network
        {'params': model.classifier.parameters(), 'lr': 1e-3}  # Higher learning rate for classifier
    ], weight_decay=1e-5)  # Weight decay for regularization
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    epochs = 30
    best_val_acc = 0.0
    patience = 10  # For early stopping
    patience_counter = 0
    
    hyperparameters = {
        'model type': 'PWC classifier - fully trainable',
        'Learning Rate for pre-trained network': 1e-5,
        'Learning rate for pwc net parameters': 1e-3,
        'Weight Decay for regularization': 1e-5,
        'Epochs': epochs,
        'Patience for early stopping': patience,
    }
    
    logger.info(f"Hyperparameters: {hyperparameters}")

    # Metrics tracking
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_labels = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch") as train_pbar:
            for (img1, img2), labels in train_pbar:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_losses.append(loss.item())
                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                train_pbar.set_postfix(loss=loss.item())
        
        # Calculate training metrics
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        
        # Validation phase
        val_losses = []
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", unit="batch") as val_pbar:
                for (img1, img2), labels in val_pbar:
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                    outputs = model(img1, img2)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())
                    val_pbar.set_postfix(loss=loss.item())
        
        # Evaluate on validation set
        val_loss = np.mean(val_losses)
        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        
        # Save metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['train_precision'].append(train_precision)
        metrics['val_precision'].append(val_precision)
        metrics['train_recall'].append(train_recall)
        metrics['val_recall'].append(val_recall)
        metrics['train_f1'].append(train_f1)
        metrics['val_f1'].append(val_f1)
        
        # Print metrics
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"  Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}")
        logger.info(f"  Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}")
        logger.info(f"  Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            {'val_acc': val_acc, 'val_precision': val_precision, 'val_recall': val_recall},
            os.path.join(output_dir, f"checkpoint_latest.pt")
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch,
                {'val_acc': val_acc, 'val_precision': val_precision, 'val_recall': val_recall},
                os.path.join(output_dir, f"checkpoint_best.pt")
            )
            logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Plot and save training curves
        plot_training_curves(metrics, os.path.join(output_dir, 'training_curves.png'))
    
    # Final evaluation on test set
    logger.info("Loading best model for final evaluation on test set...")
    checkpoint = torch.load(os.path.join(output_dir, "checkpoint_best.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
    
    logger.info("\nFinal Test Results:")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Test Precision: {test_precision:.4f}")
    logger.info(f"  Test Recall: {test_recall:.4f}")
    logger.info(f"  Test F1 Score: {test_f1:.4f}")
    
    # Save final results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
    
    logger.info(f"Training complete. All results saved to {output_dir}")

if __name__ == "__main__":
    main()