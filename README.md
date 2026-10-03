# Transfer Learning - Robot Nexus Omni 4 Roda

Proyek pembelajaran Transfer Learning untuk robot Nexus Omni 4 roda menggunakan MobileNetV2. Model contoh melakukan klasifikasi gambar menjadi dua kelas: `normal` dan `obstacle`.

## Struktur
- `dataset/train/`
- `dataset/validation/`
- `dataset/test/`
- `models/`
- `results/`
- `train.py`
- `predict.py`
- `transfer_learning.ipynb`
- `requirements.txt`

## Instalasi
```bash
pip install -r requirements.txt
```

## Dataset
Masukkan gambar nyata ke masing-masing folder `normal` dan `obstacle` pada train, validation, dan test.

## Training
```bash
python train.py
```

Model disimpan sebagai `models/nexus_mobilenetv2.keras` dan grafik training sebagai `results/training_history.png`.

## Prediksi
```bash
python predict.py --image path/to/gambar.jpg
```

## Konsep Transfer Learning
MobileNetV2 yang telah dilatih pada ImageNet digunakan sebagai feature extractor. Lapisan klasifikasi bagian akhir disesuaikan dengan kelas dataset proyek.

## Pengembangan Nexus
Model dapat dikembangkan sebagai bagian persepsi berbasis kamera pada robot Nexus Omni 4 roda untuk membantu sistem mengenali kondisi lingkungan.
