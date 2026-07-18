# Eksperimen_SML_Muhammad_Aldi_Alfatih

Proyek ini merupakan eksperimen Machine Learning dalam rangka submission Dicoding - Sistem Monitoring & Serving Machine Learning (SMSML).

## Author
**Muhammad Aldi Alfatih**  
GitHub: [AldiAlfatih](https://github.com/AldiAlfatih)

## Struktur Proyek

```
├── Membangun_model/          # Kode dan artefak pelatihan model
│   ├── modelling.py          # Model dasar dengan MLflow autolog
│   ├── modelling_tuning.py   # Model dengan hyperparameter tuning & manual logging
│   ├── requirements.txt      # Dependensi Python
│   ├── breast_cancer_preprocessing/  # Dataset preprocessing
│   ├── mlruns/               # MLflow tracking runs
│   └── training_artifacts/   # Artefak hasil pelatihan
│
├── Monitoring dan Logging/   # Monitoring & serving infrastructure
│   ├── 2.prometheus.yml              # Konfigurasi Prometheus
│   ├── 3.prometheus_exporter.py      # Prometheus metrics exporter
│   ├── 7.Inference.py                # Script inferensi
│   ├── docker-compose.yml            # Docker Compose untuk Prometheus & Grafana
│   ├── grafana_dashboard_smsml_muhammad_aldi_alfatih.json  # Dashboard Grafana
│   └── README_CARA_SCREENSHOT_VALID.md
│
└── README.md
```

## Teknologi yang Digunakan

- **Python** - Bahasa pemrograman utama
- **MLflow** - Experiment tracking dan model serving
- **Scikit-learn** - Machine learning library
- **Prometheus** - Monitoring metrics collection
- **Grafana** - Metrics visualization & alerting
- **Docker** - Containerization

## Dataset

Menggunakan dataset **Breast Cancer Wisconsin** untuk prediksi kanker payudara (klasifikasi biner).

## Cara Menjalankan

### 1. Pelatihan Model
```bash
cd Membangun_model
pip install -r requirements.txt
python modelling.py          # Basic autolog
python modelling_tuning.py   # Hyperparameter tuning
```

### 2. Serving & Monitoring
```bash
cd "Monitoring dan Logging"
pip install prometheus-client psutil requests
python 3.prometheus_exporter.py
docker compose up -d
```

Akses:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
