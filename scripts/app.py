import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import csv
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
ZONES_PRISE = ["Milieu-Droite", "Bas-Droite"]
ZONES_TRAVAIL = ["Centre", "Bas-Centre"]
ZONES_DEPOT = ["Milieu-Gauche", "Bas-Gauche"]

SEUIL_CONFIRMATION = 0.3
SEUIL_TIMEOUT_ETAPE = 15.0
DUREE_AFFICHAGE_TERMINE = 2.0
MARGIN_1_MAIN = 0.35
MAINS_REQUISES = 1

# --- Détection d'objet dans la main ---
MODEL_PATH_OBJET = "efficientdet_lite0.tflite"
SCORE_THRESHOLD_OBJET = 0.4
CLASSES_VALIDES = ["remote", "mouse"]
MARGIN_CROP_OBJET = 0.6   # marge autour de la bbox main pour chercher l'objet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_CYCLE_PATH = os.path.join(LOGS_DIR, f"cycle_log_{SESSION_ID}.txt")
LOG_MAINS_PATH = os.path.join(LOGS_DIR, f"cycle_mains_{SESSION_ID}.csv")

open(LOG_CYCLE_PATH, "w").close()

with open(LOG_MAINS_PATH, "w", newline="") as f:
    csv.writer(f).writerow(["cycle", "main1_x", "main1_y", "main2_x", "main2_y"])

# ============================================================
# MEDIAPIPE HANDS
# ============================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# MEDIAPIPE OBJECT DETECTOR (EfficientDet-Lite0)
# ============================================================
def build_object_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH_OBJET)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        score_threshold=SCORE_THRESHOLD_OBJET,
        max_results=5,
    )
    return vision.ObjectDetector.create_from_options(options)

object_detector = build_object_detector()


def zone_from_point(cx, cy, w, h):
    zone_v = "Haut" if cy < h / 3 else ("Bas" if cy > 2 * h / 3 else "Milieu")
    zone_h = "Gauche" if cx < w / 3 else ("Droite" if cx > 2 * w / 3 else "Centre")
    return "Centre" if (zone_v == "Milieu" and zone_h == "Centre") else f"{zone_v}-{zone_h}"


def get_hand_bbox(hand_landmarks, w, h):
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return (x_min, y_min, x_max, y_max), ((x_min + x_max) // 2, (y_min + y_max) // 2)


def expand_bbox(bbox, w, h, margin_ratio):
    x_min, y_min, x_max, y_max = bbox
    bw, bh = x_max - x_min, y_max - y_min
    return (max(0, int(x_min - bw * margin_ratio)),
            max(0, int(y_min - bh * margin_ratio)),
            min(w, int(x_max + bw * margin_ratio)),
            min(h, int(y_max + bh * margin_ratio)))


def union_bbox(b1, b2):
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))


def draw_grid(frame, w, h):
    cv2.line(frame, (w // 3, 0), (w // 3, h), (90, 90, 90), 1)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (90, 90, 90), 1)
    cv2.line(frame, (0, h // 3), (w, h // 3), (90, 90, 90), 1)
    cv2.line(frame, (0, 2 * h // 3), (w, 2 * h // 3), (90, 90, 90), 1)


import numpy as np

def objet_valide_dans_main(frame_rgb, bbox_main, w, h):
    x1, y1, x2, y2 = expand_bbox(bbox_main, w, h, MARGIN_CROP_OBJET)
    if x2 <= x1 or y2 <= y1:
        return False, None, None

    crop = frame_rgb[y1:y2, x1:x2]
    crop = np.ascontiguousarray(crop)   # <-- correction : force une copie contiguë en mémoire

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = object_detector.detect(mp_image)

    for detection in result.detections:
        category = detection.categories[0]
        if category.category_name in CLASSES_VALIDES:
            bb = detection.bounding_box
            bbox_objet = (x1 + bb.origin_x, y1 + bb.origin_y,
                          x1 + bb.origin_x + bb.width, y1 + bb.origin_y + bb.height)
            return True, bbox_objet, category.category_name

    return False, None, None


# ============================================================
# MACHINE À ÉTATS
# ============================================================
class MachineEtatCycle:
    def __init__(self):
        self.etat = "IDLE"
        self.t_debut_cycle = None
        self.t_debut_stable = None
        self.t_entree_etat = time.time()
        self.t_affichage_termine = None
        self.dernier_cycle = None
        self.compteur_cycle = 0

    def reset_stable(self):
        self.t_debut_stable = None

    def update(self, zone_main, nb_mains, objet_valide=False):
        now = time.time()

        if self.etat in ("PIECE_DETECTEE", "EN_COURS_TRAVAIL"):
            if now - self.t_entree_etat > SEUIL_TIMEOUT_ETAPE:
                self.etat = "IDLE"
                self.t_debut_cycle = None
                self.reset_stable()

        if self.etat == "IDLE":
            # Nouvelle contrainte : zone correcte ET objet valide dans la main
            condition_demarrage = (
                zone_main in ZONES_PRISE
                and nb_mains >= MAINS_REQUISES
                and objet_valide
            )
            if condition_demarrage:
                if self.t_debut_stable is None:
                    self.t_debut_stable = now
                elif now - self.t_debut_stable >= SEUIL_CONFIRMATION:
                    self.etat = "PIECE_DETECTEE"
                    self.t_debut_cycle = now
                    self.t_entree_etat = now
                    self.reset_stable()
            else:
                self.reset_stable()

        elif self.etat == "PIECE_DETECTEE":
            if zone_main in ZONES_TRAVAIL:
                self.etat = "EN_COURS_TRAVAIL"
                self.t_entree_etat = now

        elif self.etat == "EN_COURS_TRAVAIL":
            if zone_main in ZONES_DEPOT:
                if self.t_debut_stable is None:
                    self.t_debut_stable = now
                elif now - self.t_debut_stable >= SEUIL_CONFIRMATION:
                    self.dernier_cycle = now - self.t_debut_cycle
                    self.compteur_cycle += 1
                    self.etat = "TERMINE"
                    self.t_affichage_termine = now
                    self.reset_stable()
            else:
                self.reset_stable()

        elif self.etat == "TERMINE":
            if now - self.t_affichage_termine >= DUREE_AFFICHAGE_TERMINE:
                self.etat = "IDLE"
                self.t_debut_cycle = None

        return self.etat

    def temps_ecoule(self):
        if self.t_debut_cycle is not None and self.etat in ("PIECE_DETECTEE", "EN_COURS_TRAVAIL"):
            return time.time() - self.t_debut_cycle
        return None


def logger_cycle(numero, duree):
    ligne = f"cycle {numero} : {duree:.0f}s"
    with open(LOG_CYCLE_PATH, "a") as f:
        f.write(ligne + "\n")
    print(ligne)


def logger_mains(numero, centres_mains):
    m1 = centres_mains[0] if len(centres_mains) > 0 else ("", "")
    m2 = centres_mains[1] if len(centres_mains) > 1 else ("", "")
    with open(LOG_MAINS_PATH, "a", newline="") as f:
        csv.writer(f).writerow([numero, m1[0], m1[1], m2[0], m2[1]])


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erreur : caméra non détectée")
    exit()

machine = MachineEtatCycle()
prev_time = 0
dernieres_positions = []

print(f"Session: {SESSION_ID}")
print(f"-> {os.path.basename(LOG_CYCLE_PATH)}")
print(f"-> {os.path.basename(LOG_MAINS_PATH)}")
print("Zone de prise + objet (remote/mouse) requis pour démarrer un cycle. 'q' pour quitter")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        draw_grid(frame, w, h)

        zone_main = None
        bbox_affiche = None
        positions_courantes = []
        nb_mains = 0

        if results.multi_hand_landmarks:
            nb_mains = len(results.multi_hand_landmarks)
            hand_bboxes = []
            for hl in results.multi_hand_landmarks:
                bbox, center = get_hand_bbox(hl, w, h)
                hand_bboxes.append(bbox)
                positions_courantes.append(center)
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

            bbox_affiche = (expand_bbox(hand_bboxes[0], w, h, MARGIN_1_MAIN) if len(hand_bboxes) == 1
                             else union_bbox(hand_bboxes[0], hand_bboxes[1]))

            rcx = (bbox_affiche[0] + bbox_affiche[2]) // 2
            rcy = (bbox_affiche[1] + bbox_affiche[3]) // 2
            zone_main = zone_from_point(rcx, rcy, w, h)
            dernieres_positions = positions_courantes

            if zone_main in ZONES_PRISE:
                couleur = (0, 200, 255) if nb_mains >= MAINS_REQUISES else (0, 0, 200)
            elif zone_main in ZONES_TRAVAIL:
                couleur = (0, 255, 0)
            elif zone_main in ZONES_DEPOT:
                couleur = (255, 100, 0)
            else:
                couleur = (150, 150, 150)
            cv2.rectangle(frame, (bbox_affiche[0], bbox_affiche[1]), (bbox_affiche[2], bbox_affiche[3]), couleur, 3)
            cv2.putText(frame, zone_main, (bbox_affiche[0], bbox_affiche[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

        # ---------------- DÉTECTION D'OBJET (uniquement si pertinent) ----------------
        objet_ok = False
        objet_label = None
        bbox_objet = None

        if bbox_affiche is not None and zone_main in ZONES_PRISE and nb_mains >= MAINS_REQUISES:
            objet_ok, bbox_objet, objet_label = objet_valide_dans_main(rgb, bbox_affiche, w, h)
            if bbox_objet is not None:
                cv2.rectangle(frame, (bbox_objet[0], bbox_objet[1]), (bbox_objet[2], bbox_objet[3]),
                              (255, 0, 255), 2)
                cv2.putText(frame, objet_label, (bbox_objet[0], bbox_objet[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        etat_avant = machine.etat
        etat = machine.update(zone_main, nb_mains, objet_ok)
        if etat_avant != "TERMINE" and etat == "TERMINE":
            logger_cycle(machine.compteur_cycle, machine.dernier_cycle)
            logger_mains(machine.compteur_cycle, dernieres_positions)

        # ---------------- AFFICHAGE ----------------
        texte_etat = etat
        if etat == "TERMINE" and machine.dernier_cycle is not None:
            texte_etat = f"cycle {machine.compteur_cycle} : {machine.dernier_cycle:.0f}s"
        elif machine.temps_ecoule() is not None:
            texte_etat = f"{etat} - {machine.temps_ecoule():.1f}s"

        pos_texte = (bbox_affiche[0], max(20, bbox_affiche[1] - 35)) if bbox_affiche is not None else (10, 90)
        cv2.putText(frame, texte_etat, pos_texte, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.putText(frame, f"Mains: {nb_mains}/{MAINS_REQUISES} requises", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        texte_objet = f"Objet: {'OK (' + objet_label + ')' if objet_ok else 'absent'}"
        cv2.putText(frame, texte_objet, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Cycle de travail (main + objet requis)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    object_detector.close()