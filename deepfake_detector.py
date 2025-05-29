import os
import getpass                                  
import shutil                                   
import zipfile                                  
import time
import random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch.multiprocessing as mp


class CombinedImageDataset(Dataset):
    """
    Custom dataset that loads images from a list of (file_path, label) tuples.
    """
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def extract_zip_to_separated_folders(zip_path, out_dir):
    """
    Unzip archive.zip which has subfolders Fake/ and Real/,
    placing each image into out_dir/Fake/ or out_dir/Real/.
    """
    if os.path.exists(out_dir):
        print(f"[extract] '{out_dir}' already exists, skipping extraction.")
        return
    print(f"[extract] Unzipping {zip_path} → {out_dir}")
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for member in archive.namelist():
            parts = member.split('/')
            if len(parts) < 2 or parts[-2] not in ("Fake", "Real"):
                continue
            if not member.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            target_folder = os.path.join(out_dir, parts[-2])
            os.makedirs(target_folder, exist_ok=True)
            dest = os.path.join(target_folder, os.path.basename(member))
            with archive.open(member) as src, open(dest, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    print("Done zip extraction.")

def get_all_files(extract_dir):
    valid_exts = ('.jpg', '.jpeg', '.png')
    all_files = []                                    
    for root, _, files in os.walk(extract_dir):       
        folder = os.path.basename(root)               
        if folder not in ("Fake", "Real"):            
            continue                                  
        label = 0 if folder == "Fake" else 1          
        for file_name in files:                           
            if file_name.lower().endswith(valid_exts):    
                all_files.append((os.path.join(root, file_name), label))

    return all_files

# Main Training, Evaluation, and Visualization Function
def main():
    # 1. Parameters, path, & extraction
    BATCH_SIZE = 32             
    EPOCHS = 1                
    LEARNING_RATE = 1e-4     
    WEIGHT_DECAY = 1e-2        # Weight decay for L2 regularization
    MODEL_NAME = "google/vit-base-patch16-224"

    username = getpass.getuser()
    zip_path = f"C:/Users/{username}/Desktop/archive.zip"
    extract_dir = f"C:/Users/{username}/Desktop/extracted_archive"
    extract_zip_to_separated_folders(zip_path, extract_dir)

    all_files = get_all_files(extract_dir)

    # Randomly shuffle and split data: 70% train, 15% val, 15% test
    random.shuffle(all_files)
    total = len(all_files)
    n_train = int(0.7 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    print(f"Split: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test images.")

    # 3. Preprocessing and Data Augmentation
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # Create custom datasets
    train_dataset = CombinedImageDataset(train_files, transform=train_transform)
    val_dataset = CombinedImageDataset(val_files, transform=val_transform)
    test_dataset = CombinedImageDataset(test_files, transform=val_transform)

    # Create DataLoaders (adjust num_workers as needed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Print the assumed class mapping: Fake=0, Real=1
    print("Assumed class mapping: {'Fake': 0, 'Real': 1}")

    # 4. Initialize the Vision Transformer Model
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # 5. Optimizer, Loss Function, and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # 6. Training Function with Progress Indicator
    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss, correct = 0, 0
        for inputs, labels in tqdm(loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        return total_loss / len(loader), correct / len(loader.dataset)

    # 7. Validation Function
    def validate_epoch(model, loader, criterion):
        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(pixel_values=inputs).logits
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
        return total_loss / len(loader), correct / len(loader.dataset)

    # 8. Training Loop
    PATIENCE = 3
    best_val_loss = float("inf")
    trigger = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        if device.type == 'cuda':
            epoch_start = torch.cuda.Event(enable_timing=True)
            epoch_end = torch.cuda.Event(enable_timing=True)
            epoch_start.record()
        else:
            start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        if device.type == 'cuda':
            epoch_end.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start.elapsed_time(epoch_end) / 1000.0  # seconds
        else:
            epoch_time = time.time() - start_time

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        # Early stoppage
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
        else:
            trigger += 1
            print(f"No improvement in validation loss. Early stopping trigger is {trigger}/{PATIENCE}.")
            if trigger >= PATIENCE:
                print("Early stopping has been triggered.")
                break


    # Plot training curves
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs_range, [a*100 for a in train_accuracies], label="Train Accuracy")
    plt.plot(epochs_range, [a*100 for a in val_accuracies], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 9. Testing Function (and collect all predictions)
    def test_model(model, loader, criterion):
        model.eval()
        total_loss, correct = 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(pixel_values=inputs).logits
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return total_loss / len(loader), correct / len(loader.dataset), np.array(all_labels), np.array(all_preds)
    
    test_loss, test_acc, y_true, y_pred = test_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    print("Precision (macro): {:.4f}".format(precision))
    print("Recall (macro): {:.4f}".format(recall))
    print("F1 Score (macro): {:.4f}".format(f1))
    for i, cls in enumerate(["Fake", "Real"]):
        print(f"  {cls}: Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Fake", "Real"])
    plt.yticks(tick_marks, ["Fake", "Real"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # 10. Save the Model
    torch.save(model.state_dict(), "deepfake_vit_model.pth")
    print("Model saved as 'deepfake_vit_model.pth'")

    # 11. Display Sample Predictions from Test Set
    def display_separated_predictions(model, loader, num_each=10):
        """
        Displays sample predictions from the test set:
          - 'num_each' images predicted correctly,
          - 'num_each' images predicted incorrectly.
        In total, 20 images are displayed with:
          - the image file name,
          - predicted label (with prediction probability as percentage),
          - actual label.
        Randomly samples 100 images from the test set.
        """
        sample_indices = random.sample(range(len(loader.dataset)), min(100, len(loader.dataset)))
        correct_samples = []
        incorrect_samples = []
        for idx in sample_indices:
            img, label = loader.dataset[idx]
            file_path, _ = loader.dataset.file_list[idx]
            input_tensor = img.unsqueeze(0).to(device)
            outputs = model(pixel_values=input_tensor).logits
            pred_val = outputs.argmax(dim=1).item()
            class_map = {0: "Fake", 1: "Real"}
            pred_label = class_map[pred_val]
            actual_label = class_map[label]
            prob = outputs.softmax(dim=1)[0][pred_val].item()
            sample = (file_path, pred_label, prob, actual_label)
            if pred_label == actual_label:
                correct_samples.append(sample)
            else:
                incorrect_samples.append(sample)
        print(f"Found {len(correct_samples)} correct and {len(incorrect_samples)} incorrect predictions (from 100 sampled images).")
        num_correct = min(num_each, len(correct_samples))
        num_incorrect = min(num_each, len(incorrect_samples))
        if num_correct < num_each or num_incorrect < num_each:
            print("Not enough samples in one category; displaying available samples.")
        if num_correct > 0:
            correct_samples = random.sample(correct_samples, num_correct)
        if num_incorrect > 0:
            incorrect_samples = random.sample(incorrect_samples, num_incorrect)
        samples_to_display = correct_samples + incorrect_samples
        random.shuffle(samples_to_display)
        
        total = len(samples_to_display)
        plt.figure(figsize=(12, total * 2))
        for i, (file_path, pred_label, prob, actual_label) in enumerate(samples_to_display):
            file_name = os.path.basename(file_path)
            prob_percent = prob * 100
            img_disp = Image.open(file_path).convert("RGB")
            img_disp = transforms.Resize((224,224))(img_disp)
            plt.subplot(4, 5, i+1)
            plt.imshow(np.array(img_disp))
            plt.axis("off")
            plt.title(f"{file_name}\nPred: {pred_label} ({prob_percent:.1f}%)\nActual: {actual_label}", fontsize=8)
        plt.tight_layout()
        plt.show()

    display_separated_predictions(model, test_loader, num_each=10)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    torch.multiprocessing.freeze_support()
    main()