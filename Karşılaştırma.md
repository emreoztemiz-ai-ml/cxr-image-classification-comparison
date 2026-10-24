### Uzamsal yapı ile düzleştirilmiş yapı üzerinde toplanda 20-22 epoch'luk bir eğitim sonrası ( bahsi geçen bu fazladan 2 eğitim, veri setleri içerisine tek bir resim dahil edilerek sonuçlara etki eder bir hassasiyet kazandırıp kazandırmayacağı bahsi üzerine denendi ancak yüzdelik bağlamda sonuçlarda değişmeye sebebiyet vermediği kanısı oluştu.) alınan değerler not edilerek sizlere sunulmuştur. 

**Dataset 1: İkili Sınıflandırma (Normal vs. Pnömoni)**

| **Model**                | **Accuracy** | **Precision** | **Recall (Sens.)** | **Specificity** | **F1-Score** | **ROC-AUC** |
| ------------------------ | ------------ | ------------- | ------------------ | --------------- | ------------ | ----------- |
| **EfficientNet-B0**      | 0.97         | 0.96          | 0.98               | 0.95            | 0.97         | 0.99        |
| **ResNet-50**            | 0.96         | 0.95          | 0.97               | 0.94            | 0.96         | 0.98        |
| **Baseline CNN**         | 0.95         | 0.94          | 0.96               | 0.92            | 0.95         | 0.97        |
| **MLP (Düzleştirilmiş)** | 0.83         | 0.81          | 0.85               | 0.78            | 0.83         | 0.86        |
| **Lojistik Regresyon**   | 0.76         | 0.75          | 0.78               | 0.71            | 0.76         | 0.80        |

**Dataset 2: Çok Sınıflı Sınıflandırma (COVID-19 DB - 4 Sınıf)**

| **Model**                | **Accuracy** | **Macro-Precision** | **Macro-Recall** | **Macro-F1** | **Macro ROC-AUC** |
| ------------------------ | ------------ | ------------------- | ---------------- | ------------ | ----------------- |
| **EfficientNet-B0**      | 0.97         | 0.97                | 0.96             | 0.96         | 0.99              |
| **ResNet-50**            | 0.96         | 0.95                | 0.96             | 0.95         | 0.98              |
| **Baseline CNN**         | 0.91         | 0.90                | 0.91             | 0.90         | 0.94              |
| **MLP (Düzleştirilmiş)** | 0.70         | 0.68                | 0.69             | 0.68         | 0.75              |
| **Lojistik Regresyon**   | 0.62         | 0.58                | 0.60             | 0.59         | 0.68              |

### 2. Peki neler oldu?

**Hata Analizi ve Uzamsal Veri Kaybının Etkisi Üzerine Çıkarımlarım:**

"Projeyi kodlayıp eğitimleri tamamladığımda, teorikte öğrendiğimiz 'uzamsal verinin önemi' nin metrik tablosuna ne kadar sert yansıdığını bizzat gözlemleme fırsatım oldu.

Özellikle görev zorlaşıp Dataset 2'de 4 farklı hastalık sınıfına geçtiğimizde, beklediğim gibi klasik makine öğrenmesi modelleri ciddi anlamda zorlandı. Lojistik Regresyon ve MLP modellerinde görüntüleri bir vektör olarak dümdüz ettiğimiz için, görüntünün sağında, solunda ne olduğuna dair o mekansal bağıntı tamamen koptu. Akciğerdeki kemik yapısıyla, pnömoni lekesi model için aynı satırdaki pikseller haline geldi. Bu yüzden Lojistik Regresyon, COVID-19'un o kendine has 'buzlu cam' görünümüyle normal viral pnömoniyi birbirinden ayırt edemedi ve Macro-F1 skoru %59'lara kadar çakıldı aslında sınıflar arası karmaşıklık arttığı için yani Covid19 ile Viral Pnomöninin izleri piksel bazda çok benzediği için lojistik regresyon bu ayrımı yapamamaya başladığını düşünüyorum. Hatta kayıp grafiklerini incelediğimde, klasik modellerin belli bir hatanın altına inemediğini, bir platoya takılıp kaldığını gördüm.

Diğer tarafta ise Derin Öğrenme modellerinin (CNN, ResNet, EfficientNet) mimari üstünlüğünü çok net gördüm. Sadece kendi yazdığım basit Baseline CNN bile, filtreleri (kernel) sayesinde pikselleri düzleştirmeden o bölgesel beyazlaşmaları yakaladığı için %90'ların üzerinde başarı gösterdi.

Asıl farkı yaratan ise Transfer Learning kullandığım ResNet-50 ve EfficientNet-B0 modelleri oldu. Bu modeller ImageNet üzerinde milyonlarca resim görerek eğitildikleri için, sadece şekil bulmayı değil, dokuları ayırt etmeyi zaten biliyorlardı. ResNet'in içindeki o kalıntı bağlantıları sayesinde model ne kadar derinleşirse derinleşsin eğitim grafiği pürüzsüz bir şekilde aşağı indi. Sonuç olarak, medikal bir görüntü sınıflandırma problemi çözüyorsak, pikselleri düzleştiren yaklaşımlar yerine uzamsal hiyerarşiyi koruyan mimariler kullanmamızın bir tercih değil, zorunluluk olduğunu bu projeyle anlamış oldum.
" 
# Yazımı okuduğunuz için Teşekkür Ederim.