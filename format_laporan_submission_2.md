# Laporan Proyek Machine Learning - Dicky Candid Saragih

## Project Overview
Proyek ini bertujuan membangun sistem klasifikasi gambar makanan Padang menggunakan *deep learning*. Sistem ini akan mampu mengidentifikasi berbagai jenis makanan Padang dari gambar, serta memberikan rekomendasi makanan serupa berdasarkan kemiripan fitur visual.  Sistem ini dibangun menggunakan model *transfer learning* VGG16 yang dilatih dengan dataset gambar makanan Padang.  Prosesnya mencakup persiapan data, pembangunan model, pelatihan, evaluasi, dan demonstrasi kemampuan rekomendasi sistem.  Hasil akhir proyek diharapkan dapat memberikan solusi untuk identifikasi dan rekomendasi makanan Padang, bermanfaat bagi pengguna yang ingin mengetahui informasi lebih lanjut mengenai berbagai jenis makanan Padang.

## Business Understanding
Indonesia kaya akan kuliner, khususnya masakan Padang.  Informasi mengenai beragam jenis makanan Padang dan kemiripannya seringkali sulit didapatkan dengan cepat. Proyek ini bertujuan untuk menciptakan sebuah sistem yang dapat membantu pengguna dalam mengidentifikasi dan menemukan informasi terkait makanan Padang berdasarkan gambar. Sistem ini dapat diaplikasikan dalam berbagai platform seperti aplikasi mobile atau website, memberikan kemudahan bagi pengguna untuk mengidentifikasi makanan yang difoto dan menemukan rekomendasi makanan lain yang serupa.

### Problem Statements
pernyataan masalah:
Kurangnya informasi cepat dan akurat mengenai jenis-jenis makanan Padang dan rekomendasi makanan serupa berdasarkan gambar.  Pengguna seringkali kesulitan dalam mengidentifikasi jenis makanan Padang tertentu atau menemukan pilihan makanan serupa yang mungkin menarik bagi mereka.

### Goals
Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
Tujuan utama proyek ini adalah membangun model klasifikasi gambar yang dapat mengidentifikasi berbagai jenis makanan Padang dengan akurasi yang tinggi.  Selain klasifikasi, sistem ini juga harus dapat merekomendasikan makanan Padang lain yang serupa berdasarkan kemiripan fitur visual, sehingga pengguna dapat menemukan pilihan makanan lain yang mungkin disukai.

### Solution statements
Solusi yang diusulkan adalah membangun model klasifikasi gambar menggunakan *deep learning*, khususnya *transfer learning* dengan arsitektur VGG16.  Model ini akan dilatih dengan dataset gambar makanan Padang yang telah dikumpulkan dan diolah.  Selanjutnya, sistem akan dilengkapi dengan kemampuan rekomendasi makanan serupa berdasarkan perhitungan kemiripan fitur gambar, yang diukur menggunakan *cosine similarity*.

## Data Understanding
Dataset gambar makanan Padang diperoleh dari [dataset Kaggle]([https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data](https://www.kaggle.com/datasets/faldoae/padangfood)).. Dataset ini berisikan gambar-gambar makanan Padang dengan berbagai variasi menu. Lokasi penyimpanan dataset pada yang digunakan melalui penyimpanan google drive

Variabel-variabel makanan padang dataset adalah sebagai berikut:
- filepath: Jalur lengkap (path) menuju file gambar dalam dataset.
- label: Nama kategori makanan Padang yang menjadi label dari gambar tersebut, ayam_goreng, ayam pop, daging rendang,gulai_ikan, dendeng batokok,gulai tunjang, gulai tambusu, telur dadar, telur balado

## Data Preparation
Adapun tahapan pada data preperation sebagai berikut:
**1. Persiapan Data**
a. **Pengumpulan Data:** Data gambar makanan Padang dikumpulkan dari Google Drive dan disimpan dalam direktori `/content/drive/MyDrive/Ml_terapan/makanan_padang`. Data terbagi ke dalam beberapa kategori makanan.
b. **Pembuatan DataFrame:**  Data gambar dan label kategori diorganisir ke dalam Pandas DataFrame. DataFrame ini berisi dua kolom: `filepath` (lokasi file gambar) dan `label` (kategori makanan).
c. **Preprocessing Gambar:**
Resizing: Gambar diubah ukurannya menjadi 100x100 piksel menggunakan `cv2.resize()`.  Standarisasi ukuran ini penting untuk konsistensi input model.
Normalisasi:Nilai piksel gambar dinormalisasi ke rentang 0-1 dengan membaginya dengan 255. Normalisasi membantu meningkatkan performa model dan mencegah dominasi fitur yang memiliki nilai piksel lebih besar.
Konversi ke Array NumPy:  Daftar gambar yang telah diproses diubah menjadi array NumPy untuk digunakan dalam model.
d. **Encoding Label:** Label kategori makanan (string) diubah menjadi representasi numerik menggunakan `LabelEncoder` dari scikit-learn.  Ini diperlukan karena model machine learning umumnya bekerja dengan data numerik.
e. **Pembagian Data:** Data dibagi menjadi data latih (training) dan data uji (testing) dengan rasio 80:20 menggunakan `train_test_split`.  Stratifikasi (`stratify=y`) digunakan untuk memastikan proporsi setiap kelas makanan sama antara data latih dan data uji.
f. **Augmentasi Data:** Data augmentasi digunakan untuk meningkatkan jumlah data latih dan meningkatkan generalisasi model. `ImageDataGenerator` digunakan untuk melakukan transformasi acak pada gambar seperti rotasi, pergeseran, pemotongan, zoom, dan flip horizontal. Augmentasi membantu model belajar variasi dari gambar dan mengurangi overfitting.

## Modeling
a. **Arsitektur Model:** Digunakan model transfer learning VGG16.  `include_top=False` pada model VGG16 mengindikasikan penggunaan bagian konvolusi dari VGG16 tanpa layer klasifikasi aslinya.
b. **Transfer Learning:**  Layer-layer pada VGG16 dibekukan (`layer.trainable = False`) untuk mempertahankan bobot yang sudah dilatih sebelumnya pada dataset ImageNet. Hal ini mempercepat pelatihan dan mengurangi kebutuhan data training yang besar.  Kemudian ditambahkan beberapa layer baru di atasnya, yaitu  `Flatten`, `Dense`, `Dropout`, dan output layer dengan fungsi aktivasi softmax untuk melakukan klasifikasi sesuai kategori makanan Padang.
c. **Kompilasi Model:** Model dikompilasi menggunakan optimizer Adam, fungsi loss `sparse_categorical_crossentropy` (karena label sudah di-encode), dan metrik akurasi.
d. **Pelatihan Model:** Model dilatih menggunakan data augmentasi.  `ReduceLROnPlateau` digunakan untuk mengurangi learning rate secara otomatis jika tidak ada peningkatan pada `val_loss`, mencegah model stuck di local minimum dan meningkatkan kecepatan konvergensi.

## Evaluation
Dalam proyek ini, model klasifikasi gambar makanan Padang dibangun dengan memanfaatkan VGG16, salah satu arsitektur deep learning populer dalam transfer learning. Dengan menggunakan bobot pretrained dari ImageNet dan menyesuaikannya untuk data makanan Padang, model menunjukkan performa klasifikasi yang cukup baik pada data uji. Model mencapai akurasi sebesar 73.87%, dengan nilai presisi rata-rata 75.10%, recall 73.87%, dan f1-score 73.76%.
Beberapa kelas seperti telur_dadar dan ayam_pop memberikan performa tertinggi, masing-masing dengan f1-score 0.83 dan 0.81, menandakan bahwa fitur yang diekstraksi oleh VGG16 mampu membedakan ciri khas visual dari makanan tersebut dengan baik. Sebaliknya, makanan seperti ayam_goreng menunjukkan hasil klasifikasi yang masih lemah dengan recall sebesar 0.48, yang artinya model sering gagal mengenali kelas ini secara benar.
Secara umum, VGG16 terbukti efektif dalam mengekstraksi fitur visual dari dataset gambar makanan, namun masih terdapat peluang peningkatan melalui teknik fine-tuning lebih lanjut atau augmentasi data tambahan pada kelas yang sulit dikenali.
Evaluasi Feature Extraction dan Rekomendasi Top-N
Ekstraksi fitur dilakukan menggunakan layer akhir VGG16 (sebelum fully connected layer) yang menghasilkan representasi visual berdimensi tinggi. Fitur-fitur ini digunakan dalam sistem Content-Based Filtering untuk merekomendasikan gambar makanan yang memiliki kesamaan visual.
Melalui evaluasi manual dengan fungsi show_recommendations, sistem mampu menghasilkan rekomendasi makanan yang relevan secara visual. Misalnya, untuk input gambar dari kategori telur_balado, sistem merekomendasikan lima gambar dengan skor kemiripan (cosine similarity) tinggi, yaitu:
0.8948 – telur_balado
0.8677 – telur_balado
0.8672 – telur_balado
0.8613 – gulai_tunjang
0.8423 – gulai_tunjang

asil ini menunjukkan bahwa fitur visual yang diekstraksi dari VGG16 sangat efektif dalam mengenali kemiripan antar gambar, terutama dalam hal tekstur, warna, dan komposisi visual khas makanan. Dengan demikian, pendekatan transfer learning + cosine similarity terbukti berhasil dalam mendukung sistem rekomendasi makanan Padang berbasis konten visual.

