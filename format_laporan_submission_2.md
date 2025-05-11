# Laporan Proyek Machine Learning - Nama Anda

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
Model dievaluasi menggunakan data uji dengan beberapa metrik, antara lain:
* **Akurasi:**  [Nilai Akurasi dari output sebelumnya]
* **Presisi:** [Nilai Presisi dari output sebelumnya]
* **Recall:** [Nilai Recall dari output sebelumnya]
* **F1-Score:** [Nilai F1-Score dari output sebelumnya]

Selain metrik tersebut, *classification report* memberikan informasi lebih detail mengenai presisi, recall, dan F1-score untuk setiap kelas makanan.  Hal ini penting untuk mengidentifikasi kelas-kelas yang mungkin mengalami kesulitan dalam klasifikasi.

Sistem juga dievaluasi secara visual dengan menggunakan fungsi `show_recommendations`. Fungsi ini menampilkan gambar input dan lima rekomendasi makanan lain yang paling mirip berdasarkan *cosine similarity*.  Visualisasi ini memberikan gambaran intuitif tentang kemampuan sistem dalam merekomendasikan makanan serupa.



_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
