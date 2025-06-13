from collections import OrderedDict
import os
from pathlib import Path
from threading import Timer
import numpy as np
import torch
import yaml
import flwr as fl
from ultralytics import YOLO
from flwr.client import NumPyClient
import argparse
from huggingface_hub import snapshot_download, login
from ultralytics import YOLO
import os
import requests
import zipfile
import time

MODEL_PATH = "pretrainedModel.pt"
MODEL_URL = "https://github.com/user-attachments/files/20605529/pretrainedModel.zip"
repo_id = -1

import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import os

# .env manuell einlesen
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# Zugriff auf Variable
API_TOKEN = os.environ.get("API_TOKEN")
# Optional: Fehler anzeigen
if not API_TOKEN:
    raise RuntimeError("❌ API_TOKEN nicht gefunden! .env fehlt oder ist leer.")

print(f"🔐 Verwende Token: {API_TOKEN[:6]}...")  # gekürzt anzeigen

import socket

def get_local_ip():
    try:
        # Verbindung zu einem beliebigen externen Ziel herstellen (ohne sie tatsächlich zu senden)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS als Dummy-Ziel
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return f"❌ Fehler: {e}"

print(f"📡 Lokale IP-Adresse: {get_local_ip()}")



if not os.path.exists(MODEL_PATH):
    print("📥 Lade Modell...")
    r = requests.get(MODEL_URL)
    with open("model.zip", "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile("model.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    print("✅ Modell entpackt")

MODEL_DEF = MODEL_PATH
SERVER_ADDRESS = "192.168.1.69:8080"

def count_train_samples(data_yaml: str) -> int:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    # Neuer Pfad zu: train/images/ring_front_center/
    train_image_path = os.path.join(cfg.get("train"), "images", "ring_front_center")
    print(f"Trainpath {train_image_path}")
    if not os.path.isdir(train_image_path):
        print(f"LOCAL SAMPLES ERROR {train_image_path}")
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for fn in os.listdir(train_image_path) if os.path.splitext(fn.lower())[1] in exts)


class YOLOFlowerClient(NumPyClient):
    def __init__(self, data_yaml: str):
        super().__init__()
        self.data_yaml = data_yaml
        self.model = YOLO(MODEL_DEF)

        # Lokale Sample-Anzahl dynamisch aus Datenordner
        self.local_samples = count_train_samples(self.data_yaml)
        print(f"Local Samples {self.local_samples}")
        self.already_trained = False

    def get_parameters(self, config):
        # einfach das gecachte Modell zurückgeben
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def fit(self, parameters: list[np.ndarray], config) -> tuple[list[np.ndarray], int, dict]:
        # ─── SANITY‐CHECK VOR DEM LADEN ───
        if self.already_trained:
            print("⚠️ Kein Training durchgeführt – Rückgabe leerer Ergebnisse.")
            # Gebe die Parameter unverändert zurück, mit 0 Samples und leerem Dict
            Timer(4, lambda: os._exit(0)).start()
            return parameters, 0, {} 
        start_time = time.time()
        self.model = YOLO(MODEL_DEF)  # Statt self.model
        means_before = [float(x.mean()) for x in parameters[:3]]
        print(f"CHECK 1 [Client {Path(self.data_yaml).stem}] 🔍 Received global means (first 3 tensors): {means_before}")

        # 1) Lade globale Gewichte
        sd_keys = list(self.model.model.state_dict().keys())
        loaded = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(sd_keys, parameters)}
        )
        self.model.model.load_state_dict(loaded, strict=True)

        # ─── SANITY‐CHECK NACH DEM LADEN ───
        first_key = list(self.model.model.state_dict().keys())[0]
        post_mean = float(self.model.model.state_dict()[first_key].mean())
        print(f"CHECK 2 [Client {Path(self.data_yaml).stem}] ✅ After loading, '{first_key}' mean = {post_mean:.6f}")

        # 2) Lokales Training
        print(f"Client Training on global model")
        results = self.model.train(
            data=self.data_yaml,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,)

        train_metrics = {
            "box_loss": float(getattr(results, "box_loss", 0.0)),
            "cls_loss": float(getattr(results, "cls_loss", 0.0)),
            "train_duration": time.time() - start_time,
            "drive_id": repo_id,
            "ip": get_local_ip()
        }

        # 4) Extrahiere Updates
        updated = [v.cpu().numpy() for v in self.model.model.state_dict().values()]
        print(f"Returning Fit, {self.local_samples}, {train_metrics}")
        self.already_trained = True
        return updated, self.local_samples, train_metrics

    def evaluate(self, parameters: list[np.ndarray], config) -> tuple[float, int, dict]:
        # 1) Lade globale Gewichte
        self.model = YOLO(MODEL_DEF)  # Statt self.model
        sd_keys = list(self.model.model.state_dict().keys())
        loaded = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(sd_keys, parameters)}
        )
        self.model.model.load_state_dict(loaded, strict=True)


        print(f"Client : Running evaluation")
        results = self.model.val(
            data=self.data_yaml,
            imgsz=IMG_SIZE,
        )
        eval_metrics = results.results_dict

        # 3) Loss und Beispiele extrahieren
        loss = float(eval_metrics.get("box_loss", 0.0))
        metrics = {k: float(v) for k, v in eval_metrics.items()}

        print(f"Returning Evaluate {loss}, {self.local_samples}, {metrics}")

        # 4) Prozess kurz verzögern und beenden
        return loss, self.local_samples, metrics

# ---------- main -------------------------------------------------------------
def main() -> None:
     # Öffne die Datei im Lesemodus und lese den aktuellen Zähler.
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=YOLOFlowerClient("data.yaml")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starte den Flower-Client mit Trainingseinstellungen")
    parser.add_argument("--epochs", type=int, required=True, help="Anzahl der Epochen fürs Training")
    parser.add_argument("--img_size", type=int, required=True, help="Bildgröße fürs Training")
    parser.add_argument("--drive_id", type=int, required=True, help="Fahrt-ID")
    args = parser.parse_args()

    EPOCHS = args.epochs
    IMG_SIZE = args.img_size
    DRIVE_ID = args.drive_id

    # Drive-ID als zweistelliges Format sichern
    drive_id_str = f"{args.drive_id:02d}"  # z.B. 3 → "03"

    # Repo-Name zusammensetzen
    repo_id = f"TryhardDev/argoverse-1-trip-test-{drive_id_str}"
    print(f"📦 Lade Dataset: {repo_id}")

    login(API_TOKEN)  # nur nötig bei privaten Repos
    # Lokal herunterladen
    dataset_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",  # <<< DAS IST ENTSCHEIDEND
        local_dir=f"./datasets/argoverse_{drive_id_str}",
        local_dir_use_symlinks=False,
    )


    # Pfade für YOLO anpassen
    data_yaml = {
        "train": os.path.join(dataset_path, "train"),
        "val": os.path.join(dataset_path, "val"),
        "test": os.path.join(dataset_path, "test"),
        "nc": 8,
        "names": {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "bus",
            5: "truck",
            6: "traffic_light",
            7: "stop_sign"
        }
    }

    # Temporäre YAML-Datei schreiben
    import yaml
    with open("data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    main()