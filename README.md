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

3.  Data içerisinde yer alan URL adreslerinden ilgili veri setlerini indiriniz.
## cxr-image-classification-comparison/
└── data/
    └── dataset1/
        ├── Normal/       <-- Mendeley'deki TÜM Normal CXR resimlerini buraya atmalısınız
        └── Pneumonia/    <-- Mendeley'deki TÜM Pneumonia (Bakteri/Virüs) CXR resimlerini buraya atmanız gerekir
4.  dataset2 için Kaggle da Masks isimli klasörler ile herhangi bir gereksinimimiz bulunmamakta, bu sebeple onları görmezden gelebilirsiniz.
## cxr-image-classification-comparison/
└── data/
    └── dataset2/
        ├── COVID/             <-- Kaggle'daki COVID/Images içindeki tüm resimler
        ├── Normal/            <-- Kaggle'daki Normal/Images içindeki tüm resimler
        ├── Lung_Opacity/      <-- Kaggle'daki Lung_Opacity/Images içindeki tüm resimler
        └── Viral_Pneumonia/   <-- Kaggle'daki Viral Pneumonia/Images içindeki tüm resimler
    şeklinde yapıyı oluşturmalısınız.


