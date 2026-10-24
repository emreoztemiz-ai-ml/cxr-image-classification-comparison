import torch
import argparse
import os
import numpy as np
from src.data_loader import create_dataloaders
from src.models import BaselineCNN, get_resnet50, get_efficientnet_b0, LogisticRegressionPyTorch, MLP
from src.train import train_model
from src.evaluate import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Göğüs Radyografileri Sınıflandırma Projesi")
    parser.add_argument('--dataset', type=str, default='dataset1', choices=['dataset1', 'dataset2'])
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet', 'efficientnet', 'mlp', 'logistic'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.dataset.upper()}] Kullanılan Cihaz: {device}")
    
    data_dir = os.path.join(".", "data", args.dataset)
    if not os.path.exists(data_dir):
        print(f"DİKKAT: '{data_dir}' bulunamadı.")
        return

    print(f"'{args.dataset}' verisi yükleniyor...")
    train_loader, val_loader, class_names = create_dataloaders(data_dir, batch_size=args.batch_size, img_size=224)
    num_classes = len(class_names)
    print(f"Bulunan Sınıflar ({num_classes} adet): {class_names}")

    # Sınıf Ağırlıklarının Hesaplanması (Dengesiz veri problemi için)
    print("Sınıf ağırlıkları hesaplanıyor...")
    train_indices = train_loader.dataset.indices
    targets = [train_loader.dataset.dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    weights = total_samples / (num_classes * class_counts)
    class_weights = torch.FloatTensor(weights)
    print(f"Hesaplanan Sınıf Ağırlıkları: {weights}")

    print(f"Model başlatılıyor: {args.model.upper()}...")
    if args.model == 'cnn':
        model = BaselineCNN(num_classes)
    elif args.model == 'resnet':
        model = get_resnet50(num_classes)
    elif args.model == 'efficientnet':
        model = get_efficientnet_b0(num_classes)
    elif args.model == 'mlp':
        model = MLP(num_classes)
    elif args.model == 'logistic':
        model = LogisticRegressionPyTorch(num_classes)
    
    print("Eğitim başlıyor...")
    model, history = train_model(model, train_loader, val_loader, device, epochs=args.epochs, patience=3, class_weights=class_weights)

    print("Doğrulama seti üzerinde değerlendirme yapılıyor...")
    evaluate_model(model, val_loader, device, class_names)

if __name__ == "__main__":
    main()