import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, device, epochs=10, patience=3, class_weights=None):
    # Dengesiz veri için sınıf ağırlıklar
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Eğitim"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = corrects.double() / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping tetiklendi!")
                break
                
    model.load_state_dict(best_model_wts)
    
    # Loss Grafiğini Çizdirmke
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Eğitim ve Doğrulama Kayıp Grafiği')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    print("Kayıp grafiği 'loss_curve.png' olarak kaydedildi.")
    
    return model, history