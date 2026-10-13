# Real-Time-Hand-Gesture-Recognition-System
Embedded computer vision system for hand detection and real-time work-cycle identification (MediaPipe, OpenCV), with a monitoring dashboard (Flask). Prototyped on Raspberry Pi then migrated to Jetson Nano for improved real-time performance.

# Cycle Monitoring Vision

Pipeline de **vision par ordinateur embarquée** pour le suivi automatique des cycles d'un poste de travail industriel (**prise → travail → dépose**), avec détection des mains et de leur position (gauche, droite, haut, bas) et supervision en temps réel via un dashboard web.

Développé lors d'un stage au **Novation City** (centre d'excellence Industrie 4.0, Sousse, Tunisie) dans le cadre du cursus d'ingénieur en Électronique Industrielle à l'ENISo.

---

## Fonctionnalités

- Détection des mains avec **MediaPipe Hands**
- Détection d'objets avec **EfficientDet-Lite0** (TFLite)
- **Machine à états** du cycle : prise → travail → dépose
- Journalisation des cycles au format **JSON** (`logs/`)
- **Dashboard Flask** : flux vidéo MJPEG, boutons Start/Stop, validation manuelle des cycles


## Cibles matérielles

| Cible | Détails |
|---|---|
| Raspberry Pi 4 | Pipeline complet validé |
| Jetson Nano | Carte porteuse Auvidea JN30D, L4T R32.7.1 / JetPack 4.6.x, Ubuntu 18.04 (aarch64) |
| Caméra | Webcam USB (branchée directement, pas via un hub) |

> Inférence actuellement sur **CPU** (XNNPACK). La conversion **TensorRT** pour exploiter le GPU du Jetson est une étape à venir.

---

## Structure du dépôt

```
cycle-monitoring-vision/
├── scripts/              # Code Python du pipeline et du dashboard Flask
│   └── templates/        # Templates HTML du dashboard
    └──app.py
├── models/               # Modèles TFLite (EfficientDet-Lite0)
├── logs/                 # Logs de cycles JSON (générés, non versionnés)
├── docs/                 # Documentation, schémas, notes techniques
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Environnement conda (Miniforge)

```bash
conda create -n gesture_env python=3.11 -y
conda activate gesture_env
pip install -r requirements.txt
```

Après chaque redémarrage de la carte :

```bash
source ~/.bashrc
conda activate gesture_env
# secours : source ~/miniforge3/etc/profile.d/conda.sh
```

### 2. Spécificités Jetson Nano (JN30D)

- Versions **épinglées** dans `requirements.txt` (mediapipe 0.10.18, OpenCV 4.10.0.84, NumPy 1.26.4) : les versions plus récentes ne fonctionnent pas sur aarch64 / Ubuntu 18.04.
- Deux hooks d'activation conda permanents sont nécessaires :
  - `LD_LIBRARY_PATH` pour `libstdcxx-ng`
  - `LD_PRELOAD` pour `libgomp.so.1` (problème de bloc TLS statique sur aarch64)
- **Ne pas** réinstaller le méta-paquet `nvidia-jetpack` (trop volumineux pour l'eMMC de 14 Go, inutile pour ce pipeline).
- Utiliser le BSP **Auvidea** (et non l'image SD officielle NVIDIA) pour la JN30D.
- Caméra CSI : nécessite un pipeline GStreamer (`nvarguscamerasrc`) et non `cv2.VideoCapture(0)`.
- Réseau via ICS Windows (PC `192.168.137.1`, Jetson `192.168.137.50`) : la case ICS se décoche après la mise en veille, à réactiver manuellement.

### 3. Spécificités Raspberry Pi

```bash
sudo apt update          # ne PAS faire "apt upgrade" en cours de projet
sudo apt install libopenblas-dev   # remplace libatlas-base-dev sur Bookworm
```

- Arrêt propre : `sudo shutdown -h now` (jamais de débranchement direct, risque de corruption de la carte SD).
- Si la caméra ne se rouvre pas au second lancement : `pkill -9 -f app.py` ou redémarrer.

---

## Utilisation

```bash
conda activate gesture_env
cd scripts
python app.py
```

Puis ouvrir le dashboard : `http://<ip-de-la-carte>:5000`

> Adapter le nom du script et le port si ton point d'entrée est différent.

---

## Notes techniques

- `np.ascontiguousarray(crop)` est requis avant de construire un `mp.Image()` à partir d'une tranche NumPy.
- MediaPipe sous Linux utilise le délégué CPU (XNNPACK) par défaut : le GPU CUDA n'est pas utilisé automatiquement.
- Entraînement MobileNetV2 (Colab) : learning rate bas (1e-6), ~15 couches dégelées, rechargement explicite du meilleur checkpoint global entre les deux phases pour éviter l'oubli catastrophique.


## Auteur

**Sahar Ghazel** — Ingénierie en Électronique Industrielle, ENISo (École Nationale d'Ingénieurs de Sousse) 


