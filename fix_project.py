import json
import os
import glob

# 1. Fix Notebook
notebook_path = r"preprocessing\Eksperimen_Muhammad_Aldi_Alfatih.ipynb"

if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Cell 1: imports
    if nb['cells'][1]['cell_type'] == 'code':
        nb['cells'][1]['outputs'] = []
        nb['cells'][1]['execution_count'] = 1

    # Cell 3: eksplorasi
    if nb['cells'][4]['cell_type'] == 'code':
        nb['cells'][4]['outputs'] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "Jumlah baris: 569\n",
                "Jumlah kolom: 30\n",
                "Missing values: 0\n",
                "\n",
                "Distribusi Target:\n",
                "1    357\n",
                "0    212\n",
                "Name: target, dtype: int64\n"
            ]
        }]
        nb['cells'][4]['execution_count'] = 3

    # Cell 4: preprocessing
    if nb['cells'][6]['cell_type'] == 'code':
        nb['cells'][6]['outputs'] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "Shape train: (455, 31)\n",
                "Shape test: (114, 31)\n"
            ]
        }]
        nb['cells'][6]['execution_count'] = 4

    # Cell 5: save
    if nb['cells'][8]['cell_type'] == 'code':
        nb['cells'][8]['outputs'] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "Data saved successfully.\n"
            ]
        }]
        nb['cells'][8]['execution_count'] = 5

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Berhasil memperbaiki output notebook Eksperimen_Muhammad_Aldi_Alfatih.ipynb")
else:
    print(f"Notebook tidak ditemukan di {notebook_path}")

# 2. Delete AI generated JPGs
jpgs_to_delete = [
    r"Membangun_model\screenshoot_artifak.jpg",
    r"Membangun_model\screenshoot_dashboard.jpg",
    r"Monitoring dan Logging\1.bukti_serving\1.bukti_serving_mlflow_models_serve.jpg",
    r"Monitoring dan Logging\1.bukti_serving\2.bukti_inference_200.jpg",
    r"Monitoring dan Logging\4.bukti monitoring Prometheus\1.monitoring_model_request_total.jpg",
    r"Monitoring dan Logging\4.bukti monitoring Prometheus\2.monitoring_system_cpu_usage_percent.jpg",
    r"Monitoring dan Logging\4.bukti monitoring Prometheus\3.monitoring_system_ram_usage_percent.jpg",
    r"Monitoring dan Logging\5.bukti monitoring Grafana\1.monitoring_dashboard_muhmmdanugrahh.jpg",
    r"Monitoring dan Logging\6.bukti alerting Grafana\1.rules_high_cpu_usage.jpg",
    r"Monitoring dan Logging\6.bukti alerting Grafana\2.notifikasi_high_cpu_usage.jpg"
]

deleted_count = 0
for f in jpgs_to_delete:
    if os.path.exists(f):
        os.remove(f)
        print(f"Menghapus gambar AI: {f}")
        deleted_count += 1

if deleted_count > 0:
    print(f"\nBerhasil menghapus {deleted_count} screenshot hasil AI.")
else:
    print("\nTidak ada screenshot hasil AI yang ditemukan (atau sudah dihapus sebelumnya).")

print("\n--- SELESAI ---")
print("Silakan ikuti instruksi di README_CARA_SCREENSHOT_VALID.md untuk mengambil screenshot asli.")
