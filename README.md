ÖNEMLİ UYARI BU UYARIYI OKUMADAN DEPOYU KLONLAMAYINIZ!!!!!!
Bu proje, makine öğrenmesi ve derin öğrenme yaklaşımlarının göğüs radyografilerinden görüntü sınıflandırma görevindeki performanslarını karşılaştırmak amacıyla geliştirilmiştir.

**Uyarı:** Bu proje eğitim ve araştırma amaçlıdır. Elde edilen sonuçlar klinik olarak doğrulanmadan hastalık teşhisi veya tıbbi karar verme amacıyla kullanılmamalıdır.

# Göğüs Radyografilerinde Görüntü Sınıflandırması

Bu proje, göğüs röntgeni (CXR) görüntülerinden akciğer hastalıklarının tespitinde, uzamsal özellikleri kullanan derin öğrenme modelleri (CNN, ResNet-50, EfficientNet-B0) ile klasik makine öğrenmesi (MLP, Lojistik Regresyon) yaklaşımlarının performanslarını karşılaştırmaktadır.

## Kurulum ve Çalıştırma

1. Depoyu klonlayın:
```bash
git clone [https://github.com/emreoztemiz-ai-ml/cxr-image-classification-comparison.git](https://github.com/emreoztemiz-ai-ml/cxr-image-classification-comparison.git)
cd cxr-image-classification-comparison
```
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


## Deneyleri Çalıştırma (Kullanım)

Projemiz, kodun içine müdahale etmeden deneyleri kolayca yönetebilmeniz için dinamik bir terminal (CLI) altyapısına sahiptir. Aşağıdaki komutlarla farklı modelleri ve veri setlerini test edebilirsiniz.

### Temel Kullanım Şablonu

```bash
python main.py --dataset <dataset_adi> --model <model_adi> --epochs <sayi>
```

### Örnek Senaryolar

**1. İkili Sınıflandırmada CNN Modelini Test Etmek (Dataset 1 için gösterilmiştir. Dataset 2 için dataset1 yazan yere dataset2 yazınız)**

Normal ve Pneumonia sınıflarını birbirinden ayırmak için CNN modelini çalıştırabilirsiniz.

```bash
python main.py --dataset dataset1 --model cnn --epochs 10
```

**2. Uzamsal Veri Kaybının Etkisini İncelemek (Dataset 1)**

Görüntülerin düzleştirilerek klasik makine öğrenmesi yaklaşımıyla işlenmesinin sınıflandırma performansına etkisini incelemek için MLP modelini kullanabilirsiniz.

```bash
python main.py --dataset dataset1 --model mlp --epochs 10
```

**3. Çok Sınıflı Sınıflandırmada Transfer Learning Kullanmak (Dataset 2)**

COVID, Normal, Lung_Opacity ve Viral_Pneumonia sınıflarını içeren veri setinde ResNet modelini çalıştırabilirsiniz.

```bash
python main.py --dataset dataset2 --model resnet --epochs 15
```

**Desteklenen diğer modeller:** `efficientnet`, `logistic` dir. Terminalde yukarıda yer alan resnet ibareleri yerine burada bulunan kelimelerden herhangi birini ihtiyaç dahilinde kullanabilirsiniz.
öRNEK GÖRÜNTÜ:
<img width="866" height="481" alt="terminal-beklentisi" src="https://github.com/user-attachments/assets/eb11cd26-e399-4af6-8baa-b4ed5f046d49" />


## Çıktılar ve Değerlendirme Metrikleri

Eğitim süreci tamamlandığında, modelin performansını değerlendirmek ve deney sonuçlarını analiz etmek amacıyla aşağıdaki çıktılar oluşturulabilir. Dosyaların gerçekten otomatik oluşturulması, proje kodundaki uygulamaya bağlıdır.

### 1. Kayıp Grafiği (`loss_curve.png` olarak proje içinde paylaşılan resme gözatabilirsiniz.)

Eğitim süresince modelin kayıp değerlerinin nasıl değiştiğini gösterir. Eğitim ve doğrulama kayıpları incelenerek modelin öğrenme süreci ve olası aşırı öğrenme durumu değerlendirilebilir.
öRNEK GÖRÜNTÜ:
<img width="800" height="600" alt="loss_curve" src="https://github.com/user-attachments/assets/8521e682-3e40-4486-9355-2aeefc987bba" />


### 2. Karmaşıklık Matrisi (`confusion_matrix.png` olarak proje içinde paylaşılan resme göz atabilirsiniz.)

Modelin gerçek sınıflar ile tahmin ettiği sınıflar arasındaki ilişkiyi gösterir. Bu grafik sayesinde modelin hangi sınıfları doğru tahmin ettiği ve hangi hastalık sınıflarını birbiriyle karıştırdığı incelenebilir.
ÖRNEK GÖRÜNTÜ:
<img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e03dcfd6-7bf6-4850-9bc2-47be7df6ce05" />


### 3. Performans Metrikleri

Deney sonuçlarının değerlendirilmesinde aşağıdaki metriklerden yararlanılabilir:

- **Precision (Kesinlik):** Pozitif olarak tahmin edilen örneklerin ne kadarının doğru olduğunu gösterir.
- **Recall (Duyarlılık):** İlgili sınıfa ait gerçek örneklerin ne kadarının doğru tespit edildiğini gösterir.
- **F1-Score:** Precision ve Recall değerlerinin harmonik ortalamasıdır.
- **ROC-AUC:** Modelin sınıfları ayırt etme gücünü ölçer. Çok sınıflı görevlerde uygun hesaplama yöntemi kullanılmalıdır.
- **Class Weights (Sınıf Ağırlıkları):** Sınıflar arasındaki örnek sayısı farklılıklarını eğitim sırasında dengelemek için kullanılabilir.

## Projenin Amacı ve Temel Hipotezi

Bu projenin temel amacı, göğüs radyografilerinden hastalık sınıflandırması yapılırken farklı model mimarilerinin performanslarını karşılaştırmak ve görüntülerin uzamsal özelliklerinin korunmasının sınıflandırma başarısına etkisini deneysel olarak incelemektir.

Evrişimli sinir ağları (CNN), görüntülerdeki yerel örüntüleri ve uzamsal ilişkileri öğrenebilir. Buna karşılık, görüntülerin düzleştirilerek klasik makine öğrenmesi modellerine aktarılması, pikseller arasındaki iki boyutlu komşuluk bilgisinin doğrudan kullanılmasını sınırlar.

Bu kapsamda aşağıdaki araştırma soruları incelenmektedir:

1. Uzamsal özellikleri kullanan derin öğrenme modelleri ile görüntüleri düzleştirerek işleyen yaklaşımlar arasında nasıl performans farklılıkları bulunmaktadır?
2. İkili ve çok sınıflı sınıflandırma görevlerinde model başarısı nasıl değişmektedir?
3. CNN, ResNet ve EfficientNet mimarileri arasında hangi performans farklılıkları gözlemlenmektedir?
4. Sınıf dengesizliği, modelin sınıflara ait Precision, Recall ve F1-Score değerlerini nasıl etkilemektedir? (Nicel veriler üzerinde)

Deney sonuçları bu sorular doğrultusunda değerlendirilerek farklı mimarilerin güçlü ve zayıf yönleri ortaya konulacaktır. Uzamsal özelliklerin korunmasının model başarısına etkisi, deneysel sonuçlarla desteklendiği ölçüde yorumlanacaktır.

ÖNEMLİ UYARI BU UYARIYI OKUMADAN DEPOYU KLONLAMAYINIZ!!!!!!
Bu proje, makine öğrenmesi ve derin öğrenme yaklaşımlarının göğüs radyografilerinden görüntü sınıflandırma görevindeki performanslarını karşılaştırmak amacıyla geliştirilmiştir.

**Uyarı:** Bu proje eğitim ve araştırma amaçlıdır. Elde edilen sonuçlar klinik olarak doğrulanmadan hastalık teşhisi veya tıbbi karar verme amacıyla kullanılmamalıdır.
