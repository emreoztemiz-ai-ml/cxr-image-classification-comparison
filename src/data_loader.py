import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

def get_transforms(img_size=224):
    """Eğitim ve test için görüntü ön işleme (transform) adımlarını döndürür."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def create_dataloaders(data_dir, batch_size=32, img_size=224, val_split=0.2):
    """Verilen dizindeki görüntülerden eğitim, doğrulama ve test dataloader'larını oluşturur."""
    train_transform, test_transform = get_transforms(img_size)
    
    # Tüm veri setini PyTorch ImageFolder ile yükle
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    except FileNotFoundError:
        raise FileNotFoundError(f"Hata: {data_dir} dizini bulunamadı. Veri setini indirdiğinizden emin olun.")

    # Veri setini bölme (Split) işlemleri
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # Tekrar edilebilirlik için
    )
    
    # Doğrulama (Validation) seti için test transformlarını uygula (Veri sızıntısını önlemek için)
    val_dataset.dataset.transform = test_transform
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_names = full_dataset.classes
    
    return train_loader, val_loader, class_names

if __name__ == "__main__":
    # Test mekanizması
    print("Data Loader modülü başarıyla oluşturuldu. Sınıf fonksiyonları eklendi.")