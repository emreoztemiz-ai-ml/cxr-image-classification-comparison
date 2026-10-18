# Göğüs Radyografilerinde Görüntü Sınıflandırması

Bu proje, göğüs röntgeni (CXR) görüntülerinden akciğer hastalıklarının tespitinde, uzamsal özellikleri kullanan derin öğrenme modelleri (CNN, ResNet-50, EfficientNet-B0) ile klasik makine öğrenmesi (MLP, Lojistik Regresyon) yaklaşımlarının performanslarını karşılaştırmaktadır.

## Kurulum ve Çalıştırma

1. Depoyu klonlayın:
```bash
git clone [https://github.com/emreoztemiz-ai-ml/cxr-image-classification-comparison.git](https://github.com/emreoztemiz-ai-ml/cxr-image-classification-comparison.git)
cd cxr-image-classification-comparison

2. Sanal ortam oluşturun ve bağımlılıkları yükleyin:
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
pip install -r requirements.txt