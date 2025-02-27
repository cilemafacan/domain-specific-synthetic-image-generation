# TR
# Baseline Stable Diffusion ve Google Path Foundation Modelleri ile Sentetik Veri Üretimi

## 1. Proje Tanımı ve Amaçlar
Bu proje, Google Path Foundation modelini kullanarak histopatoloji görüntülerinden anlamlı embedding vektörleri üretmeyi ve bu vektörlerle sentetik veri üretimini amaçlamaktadır. Üç ana çalışma gerçekleştirilmiştir:

1. **Text ile Veri Üretimi**: Klasik bir Stable Diffusion mimarisi text bilgisi ile eğitilmiştir. Eğitim sonucunda text embedding kullanılarak 256x256 boyutunda histopatoloji görüntüleri üretilmiştir.

![Text2Image Mimarisi](./data/t2i.png)

2. **Image ile Veri Üretimi**: Klasik bir Stable Diffusion mimarisi image bilgisi ile eğitilmiştir. Eğitim sonucunda image embedding kullanılarak 256x256 boyutunda histopatoloji görüntüleri üretilmiştir.

![Image2Image GAN Mimarisi](./data/i2i.png)

3. **Google Path Embedding ile Veri Üretimi**: Google Path Foundation modelinden üretilen embedding vektörlerini kullanarak bir Stable Diffusion modeli eğitilmiştir. Eğitim sonucunda bu embedding vektörleri ile 256x256 boyutunda histopatoloji görüntüleri üretilmiştir.

![Google Path Embedding ile Veri Üretimi](./data/e2i.png)

## 2. Metodoloji

### 2.1 Veri Seti
- **Kaynak**: [HuggingFace](https://huggingface.co/datasets/Cilem/mixed-histopathology-512) veri seti kullanılmıştır. Bu veriler 512x512 boyutlarında patchler halinde hazırlanmıştır.
- **Görsellerin Boyutu**: 512x512 -> 256x256 yeniden boyutlandırma işlemi yapılmıştır.

![Eğitim Veri Seti Örnekleri](./data/training_imgs.png)
- **Veri Dönüşümleri**:
  - `Resize(size=256)`
  - `ToTensor()`
  - `Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`

### 2.2 Hiperparametreler
| Parametre                 | Text2Image Değeri | Image2Image ile GAN Değeri | GoogleEmbedding2Image Değeri |
|---------------------------|--------------|-----------------------| ---------------- |
| Batch Boyutu              | 32          | 32                   | 32             |
| Görüntü Boyutu            | 256          | 256                   | 256             |
| Kanal Sayısı              | 3            | 3                     | 3         |
| Embedding Vektör Boyutu   | num_tokens x 768          | 768                   | 384             |
| Epoch Sayısı              | 10           | 10                    | 10             |
| Öğrenme Hızı              | 1e-08      | 1e-08               | 1e-08           |

## 3. Sonuçlar ve Gözlemler

### 3.1 Kayıp Fonksiyonu İlerlemesi
Eğitim süresince kaydedilen loss değerleri aşağıda verilmiştir:

- Text2Image için kayıp fonksiyonlarının ilerlemesi.
    * Epoch 1/10, Loss: 0.21289968490600586
    * Epoch 2/10, Loss: 0.25580060482025146
    * Epoch 3/10, Loss: 0.1656205803155899
    * Epoch 4/10, Loss: 0.1453578770160675
    * Epoch 5/10, Loss: 0.15953485667705536
    * Epoch 6/10, Loss: 0.19383955001831055
    * Epoch 7/10, Loss: 0.10896158218383789
    * Epoch 8/10, Loss: 0.20812615752220154
    * Epoch 9/10, Loss: 0.15635214745998383
    * Epoch 10/10, Loss: 0.1979628950357437

- Image2Image için kayıp fonksiyonlarının ilerlemesi.
    * Epoch 1/10, Loss: 0.18531584739685059
    * Epoch 2/10, Loss: 0.31894707679748535
    * Epoch 3/10, Loss: 0.13547834753990173
    * Epoch 4/10, Loss: 0.22493816912174225
    * Epoch 5/10, Loss: 0.08720849454402924
    * Epoch 6/10, Loss: 0.16207090020179749
    * Epoch 7/10, Loss: 0.1695273220539093
    * Epoch 8/10, Loss: 0.15867063403129578
    * Epoch 9/10, Loss: 0.07108891010284424
    * Epoch 10/10, Loss: 0.20752660930156708

- GoogleEmbedding2Image için kayıp fonksiyonlarının ilerlemesi.

    * Epoch 1/10, Loss: 0.030875496566295624
    * Epoch 2/10, Loss: 0.18258614838123322
    * Epoch 3/10, Loss: 0.2437487691640854
    * Epoch 4/10, Loss: 0.16098453104496002
    * Epoch 5/10, Loss: 0.14515283703804016
    * Epoch 6/10, Loss: 0.16885484755039215
    * Epoch 7/10, Loss: 0.13852852582931519
    * Epoch 8/10, Loss: 0.1334361433982849
    * Epoch 9/10, Loss: 0.1788838654756546
    * Epoch 10/10, Loss: 0.11216995120048523


### 3.2 FID Sonuçları
- Text2Image için FID ve IS değerleri:
    * FID: 415.36
  
- Image2Image için FID ve IS değerleri:
    * FID: 162.16
   
- GoogleEmbedding2Image için FID ve IS değerleri:
    * FID: 62.18


## 3.3 TSNE ile Görselleştirme

- Text2Image için TSNE sonucu:

![Text2Image TSNE](./data/tsne_text2image.png)

- Image2Image için TSNE sonucu:

![Image2Image TSNE](./data/tsne_image2image.png)

- GoogleEmbedding2Image için TSNE sonucu:

![GoogleEmbedding2Image TSNE](./data/tsne_embedding2image.png)

## 3.4 Örnek Sentetik Görüntüler

- Text2Image için örnek sentetik görüntüler:

![Text2Image Örnek Görüntüler](./data/generated_images_t2i.png)

- Image2Image için örnek sentetik görüntüler:

![Image2Image Örnek Görüntüler](./data/generated_images_i2i.png)

- GoogleEmbedding2Image için örnek sentetik görüntüler:

![GoogleEmbedding2Image Örnek Görüntüler](./data/generated_images_e2i.png)

