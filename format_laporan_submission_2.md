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
Dataset gambar makanan Padang diperoleh dari [dataset Kaggle](https://www.kaggle.com/datasets/faldoae/padangfood) Dataset ini berisikan gambar-gambar makanan Padang dengan berbagai variasi menu. Lokasi penyimpanan dataset pada yang digunakan melalui penyimpanan google drive

Variabel-variabel makanan padang dataset adalah sebagai berikut:
filepath: Jalur lengkap (path) menuju file gambar dalam dataset.
jumlah gambar : 993 foto
label: Nama kategori makanan Padang yang menjadi label dari gambar tersebut, ayam_goreng, ayam pop, daging rendang,gulai_ikan, dendeng batokok,gulai tunjang, gulai tambusu, telur dadar, telur balado
Kondisi data: tidak ada duplikat

## Data Preparation
**1. Pra-pemrosesan Data:**
Tahapan ini meliputi pembacaan data gambar dan label kategori makanan dari direktori penyimpanan di Google Drive.  Selanjutnya, dilakukan:
a. **Pembentukan DataFrame:** Data gambar dan label diorganisir dalam pandas DataFrame, memudahkan pengelolaan dan akses data.
b. **Resizing dan Normalisasi Gambar:** Gambar diubah ukurannya menjadi 100x100 piksel dan dinormalisasi ke rentang 0-1.  Proses ini penting untuk keseragaman input model dan efisiensi komputasi, walau berpotensi menghilangkan detail halus gambar.  Sebagai perbaikan, perlu dieksplorasi metode *resizing* alternatif seperti padding atau cropping.

**2. Pembagian Data:**
Data dibagi menjadi data latih dan data uji dengan rasio 80:20. Metode stratifikasi digunakan untuk memastikan proporsi kelas tetap sama di kedua dataset, mengurangi bias dalam model.  Penggunaan random_state memungkinkan reproduksibilitas hasil.

**3. Augmentasi Data:**
Augmentasi data dilakukan pada data latih menggunakan ImageDataGenerator.  Teknik yang diterapkan termasuk rotasi, pergeseran, shear, zoom, dan flip horizontal. Tujuannya untuk meningkatkan variabilitas data latih dan meningkatkan generalisasi model.  Perlu dievaluasi parameter augmentasi dan metode fill_mode untuk optimalisasi.

**4. Pelatihan Model:**
Model yang digunakan adalah VGG16, memanfaatkan *transfer learning* dengan membekukan sebagian lapisan awal model dan menambahkan lapisan klasifikasi baru.  Metode *transfer learning* ini membantu meningkatkan akurasi dan efisiensi pelatihan. Penggunaan *optimizer Adam*, *loss function sparse_categorical_crossentropy*, dan metrik akurasi digunakan untuk proses pelatihan.  *ReduceLROnPlateau* diterapkan untuk penyesuaian *learning rate* selama pelatihan.

**5. Evaluasi Model:**
Model dievaluasi menggunakan data uji dengan metrik akurasi, presisi, *recall*, dan *F1-score*.  Confusion matrix digunakan untuk visualisasi kinerja model dalam mengklasifikasikan setiap kategori makanan.  Metrik evaluasi menunjukkan kinerja model dalam hal akurasi dan kemampuan generalisasi pada data yang belum pernah dilihat.

**6. Ekstraksi Fitur dan Sistem Rekomendasi:**
Fitur gambar diekstrak dari lapisan kedua-terakhir model VGG16 yang telah dilatih.  *Cosine similarity* digunakan untuk menghitung tingkat kemiripan antara vektor fitur gambar. Sistem rekomendasi memberikan *top-N* rekomendasi gambar dengan kemiripan visual tertinggi dengan gambar input.


## Modeling
**1. Ekstraksi Fitur Gambar:**
Fitur gambar diekstrak menggunakan model VGG16 yang telah dilatih sebelumnya (*pre-trained*) pada dataset ImageNet.  Model VGG16 dimodifikasi dengan menambahkan lapisan klasifikasi baru di atasnya, dan lapisan-lapisan awal dibekukan untuk memanfaatkan pengetahuan yang sudah ada.  Fitur diekstrak dari lapisan kedua terakhir model yang telah dimodifikasi. Proses *resizing* gambar dilakukan dengan ukuran tetap (100x100 piksel), yang berpotensi menghilangkan detail gambar.  Normalisasi piksel dilakukan ke rentang 0-1.

**2. Perhitungan Kemiripan (Cosine Similarity):**
Kemiripan antar gambar dihitung menggunakan *cosine similarity* pada vektor fitur yang telah diekstrak.  Nilai *cosine similarity* menunjukkan seberapa dekat dua vektor fitur dalam ruang fitur, yang merepresentasikan kemiripan visual antara dua gambar makanan.

**3. Pembuatan Rekomendasi (Top-N Recommendation):**
Sistem rekomendasi menghasilkan daftar *Top-N* rekomendasi makanan berdasarkan nilai *cosine similarity*.  Untuk setiap gambar masukan, sistem mengidentifikasi *N* gambar lain dengan nilai *cosine similarity* tertinggi.  Pada contoh implementasi, digunakan *top-5* rekomendasi (N=5).

**4. Evaluasi Kualitas Rekomendasi:**
Evaluasi dilakukan secara manual dengan menampilkan gambar input dan *top-5* rekomendasi.  Evaluasi ini dilakukan dengan visualisasi gambar masukan dan gambar rekomendasi, dan ditampilkan beserta label kategori masing-masing.  Metode evaluasi ini bertujuan untuk mengamati secara langsung kesamaan visual antara gambar masukan dan gambar rekomendasi.  Model klasifikasi gambar mencapai akurasi 74%, serta  presisi, recall, dan skor F1 yang sebanding. *Confusion matrix* digunakan untuk analisis lebih lanjut mengenai kinerja klasifikasi untuk setiap kategori.

**5. Analisis Kesesuaian Rekomendasi dengan Input:**
Berdasarkan hasil visualisasi rekomendasi, dapat dilakukan analisis kesesuaian secara manual.  Analisis ini mengamati seberapa tepat rekomendasi yang diberikan berdasarkan kemiripan visual dengan gambar masukan.  Kualitas rekomendasi dipengaruhi oleh kualitas ekstraksi fitur dan metode perhitungan kemiripan yang digunakan.

**6. Visualisasi Hasil Rekomendasi:**
Hasil rekomendasi divisualisasikan dengan menampilkan gambar input dan *top-N* rekomendasi, disertai label kategori masing-masing. Visualisasi ini memudahkan evaluasi manual terhadap kesesuaian rekomendasi dengan input.


## Evaluation
Dalam proyek ini, model klasifikasi gambar makanan Padang dibangun dengan memanfaatkan VGG16, salah satu arsitektur deep learning populer dalam transfer learning. Dengan menggunakan bobot pretrained dari ImageNet dan menyesuaikannya untuk data makanan Padang, model menunjukkan performa klasifikasi yang cukup baik pada data uji. Model mencapai akurasi sebesar 73.87%, dengan nilai presisi rata-rata 75.10%, recall 73.87%, dan f1-score 73.76%. 
**Gambar hasil evaluasi menggunakan hasil cosine similarity** 
![Gambar 1](https://github.com/user-attachments/assets/89c601ad-0f7b-4600-adf8-73a1e63a006b)
![Gambar 2](https://github.com/user-attachments/assets/757c0619-d416-4e91-b41d-99d611dc7347)

- Deskripsi Eksperimen
Dalam eksperimen ini, dikembangkan sebuah sistem rekomendasi untuk makanan khas Padang berbasis citra visual. Sistem dirancang untuk memberikan lima rekomendasi teratas berdasarkan kemiripan visual dengan gambar input. Pendekatan yang digunakan adalah content-based filtering dengan perhitungan cosine similarity, yang diperoleh dari hasil ekstraksi fitur visual menggunakan model CNN VGG-16.

Langkah-Langkah Proses Evaluasi
- Input Gambar
Gambar makanan “telur balado” digunakan sebagai input utama dalam proses rekomendasi.

- Ekstraksi Fitur
Fitur dari citra input serta seluruh citra dalam basis data diekstraksi menggunakan model CNN (VGG-16) untuk menghasilkan representasi dalam bentuk vektor fitur.

- Perhitungan Kemiripan
Dilakukan perhitungan cosine similarity antara vektor fitur gambar input dengan seluruh vektor pada dataset.

- Penyusunan Rekomendasi
Lima gambar dengan nilai cosine similarity tertinggi dipilih sebagai hasil rekomendasi.

- Evaluasi Visual dan Semantik
Evaluasi dilakukan berdasarkan kesesuaian visual serta kecocokan semantik terhadap kategori makanan.

Hasil Rekomendasi 

| No | Index Gambar | Skor Similarity | Kategori Makanan  |
|----|--------------|------------------|--------------------|
| 1  | 100          | 0.9274           | telur_balado       |
| 2  | 10           | 0.8959           | telur_balado       |
| 3  | 104          | 0.8936           | telur_balado       |
| 4  | 59           | 0.8910           | telur_balado       |
| 5  | 79           | 0.8844           | telur_balado       |


- Evaluasi dan Analisis
Tingkat Akurasi Visual
Berdasarkan hasil rekomendasi di atas, sistem berhasil memberikan lima rekomendasi teratas yang seluruhnya termasuk dalam kategori "telur balado". Hal ini menunjukkan bahwa pendekatan cosine similarity mampu mengenali kesamaan visual dengan baik, terutama dari warna dominan (merah) dan tekstur makanan.

-Kesesuaian Semantik
Untuk gambar input lainnya (tidak ditampilkan di sini), terdapat beberapa hasil rekomendasi yang termasuk makanan berbeda seperti daging rendang atau gulai tunjang. Meski berbeda secara kategori makanan, visual dari makanan tersebut memiliki kemiripan warna dan tekstur, sehingga masuk ke dalam hasil rekomendasi.

- Kesalahan Minor
  
Munculnya makanan selain "telur balado" dalam sebagian rekomendasi dapat dikategorikan sebagai kesalahan minor yang disebabkan oleh kemiripan fitur visual, bukan oleh kesalahan ekstraksi fitur.

- Kesimpulan
  
Sistem rekomendasi yang dikembangkan mampu memberikan hasil yang cukup relevan berdasarkan kemiripan visual. Dengan pendekatan content-based filtering berbasis cosine similarity, sistem dapat mengenali dan merekomendasikan makanan dengan tampilan serupa. Walaupun masih terdapat kesalahan minor dari sisi semantik, secara umum sistem dapat mendukung pengguna dalam menemukan makanan dengan visual yang mirip dan menggugah selera.





