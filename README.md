# Heart Disease — Faz 1 (Uygulamalı Yapay Zekâ)

**Yağmur Atmaca | Konya Teknik Üniversitesi**

Kalp hastalığı etiketinin tablosal sınıflandırılması için üç gerçek UCI veri setinin **keşifsel analizi, eksik değer incelemesi, tam kayıt örtüşmesi ve eğitim verisine özel ön işleme** çalışmasıdır. Faz 1'de model eğitimi veya klinik değerlendirme yapılmaz.

## İçerik
- `veri/processed.cleveland.data` (303 ham ve tekil gözlem)
- `veri/heart.dat` (270 ham ve tekil gözlem)
- `veri/processed.hungarian.data` (294 ham / 293 tekil gözlem)
- `veri/KAYNAK.md`: resmî UCI DOI, açık veri aynası ve dosya bütünlüğü.
- `faz1_analiz.py`: doğrulanabilir veri okuma, EDA, kayıt örtüşmesi ve eğitim-katı ön işleme.
- `notebook/Faz1_Kalp_Hastaligi.ipynb`: hücreli Jupyter Notebook; çıktı üretmek için Run All.
- `requirements.txt`: Python bağımlılıkları.
- `testler/test_faz1.py`: betikten üretilen çıktı ve grafik testleri.
- `ciktilar/faz1_dogrulama_ozeti.json`: yerel olarak doğrulanan temel sonuçların kaydı.

## Kurulum ve çalıştırma
```bash
python -m pip install -r requirements.txt
python faz1_analiz.py
python -m unittest discover -s testler -v
jupyter lab
```

Notebook'u depo kökünden açın. Çalıştırıldığında **7 PNG grafik** ve CSV/JSON çıktılarını üretir. İnternetten veri indirilmez; veri dosyaları depoda vardır.

## Doğrulanmış Faz 1 sayıları
| Ölçüm | Değer |
|---|---:|
| Cleveland ham/tekil | 303/303 |
| Statlog ham/tekil | 270/270 |
| Hungarian ham/tekil | 294/293 |
| Satır toplamı | 866 |
| Farklı tam kayıt örüntüsü | 596 |
| Cleveland–Statlog eşleşmesi | 270/270 |
| Cleveland–Hungarian eşleşmesi | 0/293 |
| Hungarian eksik hücre (13 alan) | 778 |
| Hungarian eksik hücre (10 ortak alan) | 34 |
| Cleveland ilk-kat tam eğitim matrisi | 242×28 |
| Cleveland ilk-kat ortak eğitim matrisi | 242×18 |
| Hungarian dış dönüşüm matrisi | 293×18 |

**Önemli:** 596 kayıt örüntüsü, 596 bağımsız hasta demek değildir. Statlog tümüyle Cleveland altkümesidir; bağımsız dış test olarak ele alınamaz. Hungarian'da tam eşleşme yoktur ama hasta kimlikleri bulunmadığından kişi düzeyinde bağımsızlık ispatlanamaz. Bu proje klinik tanı aracı değildir.

## Faz 2'de yapılacaklar
Beş aday modelin (LR, RBF-SVM, RF, KNN, yüksek kapasiteli MLP) eğitim katlarında hiperparametre seçimi, nested CV, kalibrasyon, ROC-AUC/F1/Brier ve Hungarian üzerinde sınama **henüz yapılmamıştır**.

**Veri kaynakları:** UCI Heart Disease DOI 10.24432/C52P4X; UCI Statlog Heart DOI 10.24432/C57303.

Sürüm kontrolü: dosyalar proje geliştikçe anlamlı, gerçek tarihli commitlerle eklenmiştir; önceki tek başlangıç commit'i geçmişte yapılmamış çalışma kanıtı olarak sunulmaz.
