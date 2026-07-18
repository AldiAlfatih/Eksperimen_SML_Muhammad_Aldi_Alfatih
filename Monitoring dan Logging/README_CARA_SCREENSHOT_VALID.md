# Cara mengambil screenshot valid Kriteria 4

Jalankan exporter:

```bash
pip install prometheus-client psutil requests
python 3.prometheus_exporter.py
```

Jalankan Prometheus dan Grafana:

```bash
docker compose up -d
```

Buka Prometheus `http://localhost:9090` dan query minimal:
- model_request_total
- system_cpu_usage_percent
- system_ram_usage_percent

Buka Grafana `http://localhost:3000`, import `grafana_dashboard_smsml_muhammad_aldi_alfatih.json`, pastikan dashboard bernama `dashboard-muhammad-aldi-alfatih`.
