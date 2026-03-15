import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

# -----------------------
# CONSTANTS
# -----------------------
IMG_SIZE = 320
MAX_AGE = 240.0

# -----------------------
# Project root
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "DATASET"
MODEL_PATH = PROJECT_ROOT / "models" / "bone_age_final_model.keras"

ART_DIR_CANDIDATES = [
    DATASET_DIR / "Articular_Surface_Test"
]
EPI_DIR_CANDIDATES = [
    DATASET_DIR / "Epiphysis_Test"
]
CSV_CANDIDATES = [
    DATASET_DIR / "test.csv"
]
FULL_IMG_DIR = DATASET_DIR / "Tam_Görüntüler"

def load_img(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image could not be read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype(np.float32)

def find_pair_paths(sample_id: str) -> tuple[Path, Path] | None:
    """Search ID.png files in art/epi folders."""
    fname = f"{sample_id}.png"
    art_path = None
    epi_path = None

    for d in ART_DIR_CANDIDATES:
        p = d / fname
        if p.exists():
            art_path = p
            break

    for d in EPI_DIR_CANDIDATES:
        p = d / fname
        if p.exists():
            epi_path = p
            break

    if art_path and epi_path:
        return art_path, epi_path
    return None

def build_id_to_meta_map() -> dict[str, dict]:
    """
    From train/val/test CSV files:
    id -> {"male": 0/1, "boneage": float}
    """
    mapping: dict[str, dict] = {}
    for csv_path in CSV_CANDIDATES:
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if not {"id", "male", "boneage"}.issubset(df.columns):
            continue

        for _, row in df.iterrows():
            sid = str(row["id"])
            mapping[sid] = {
                "male": 1.0 if bool(row["male"]) else 0.0,
                "boneage": float(row["boneage"]),
            }
    return mapping

class BoneAgeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bone Age Predictor (Offline)")
        self.resize(520, 180)

        # --- model ---
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found: {MODEL_PATH}")

        self.model = load_model(str(MODEL_PATH), compile=False)
        self.id_to_meta = build_id_to_meta_map()

        # --- UI ---
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(18)

        title = QLabel("BONE AGE QUERY SCREEN")
        title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 800;")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.id_input = QLineEdit()
        self.id_input.setFixedWidth(300)
        self.id_input.setPlaceholderText("Enter ID")
        self.id_input.setFixedHeight(44)
        self.id_input.setStyleSheet("font-size: 16px; padding: 8px;")

        self.btn = QPushButton("Query")
        self.btn.setFixedHeight(44)
        self.btn.setFixedWidth(140)
        self.btn.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.btn.clicked.connect(self.on_predict)

        row.addWidget(self.id_input)
        row.addWidget(self.btn)
        row.addStretch(1)

        self.sex_lbl = QLabel("Sex: -")
        self.true_lbl = QLabel("Age: -")
        self.pred_lbl = QLabel("Predicted Bone Age: -")

        for lab in (self.sex_lbl, self.true_lbl, self.pred_lbl):
            lab.setStyleSheet("font-size: 22px; font-weight: 700;")

        # Right side: image
        self.image_lbl = QLabel("IMAGE")
        self.image_lbl.setFixedSize(900, 700)
        self.image_lbl.setAlignment(Qt.AlignCenter)
        self.image_lbl.setStyleSheet("border: 1px solid #444; background-color: #000000; color: #ffffff; font-size: 15px; font-weight: bold;")

        content_row = QHBoxLayout()

        # Left: information
        info_col = QVBoxLayout()
        info_col.addLayout(row)
        info_col.addSpacing(20)
        info_col.addWidget(self.sex_lbl)
        info_col.addWidget(self.true_lbl)
        info_col.addWidget(self.pred_lbl)
        info_col.addStretch(1)

        content_row.addLayout(info_col, 1)
        content_row.addWidget(self.image_lbl, 0)

        layout.addLayout(content_row)
        layout.addStretch(1)

        self.setLayout(layout)

        self.id_input.returnPressed.connect(self.on_predict)

    def on_predict(self):
        sample_id = self.id_input.text().strip()
        if not sample_id:
            QMessageBox.warning(self, "Warning", "Please enter an ID.")
            return

        pair = find_pair_paths(sample_id)
        if pair is None:
            QMessageBox.critical(
                self,
                "Not Found",
                "No record exists with this ID."
            )
            return
        
        art_path, epi_path = pair

        if sample_id not in self.id_to_meta:
            QMessageBox.critical(
                self,
                "Record Not Found",
                "No record exists with this ID."
            )
            return

        meta = self.id_to_meta[sample_id]
        male = float(meta["male"])
        true_age = float(meta["boneage"])

        sex_text = "Male" if male >= 0.5 else "Female"
        self.sex_lbl.setText(f"Sex: {sex_text}")
        self.true_lbl.setText(f"Age: {true_age:.0f} months")

        try:
            art = load_img(art_path)
            epi = load_img(epi_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        X_art = np.expand_dims(art, axis=0)
        X_epi = np.expand_dims(epi, axis=0)
        X_male = np.array([[male]], dtype=np.float32)

        pred = self.model.predict(
            {"art_input": X_art, "epi_input": X_epi, "male_input": X_male},
            verbose=0
        ).reshape(-1)[0]

        pred_months = float(pred) * MAX_AGE
        self.pred_lbl.setText(f"Predicted Bone Age: {pred_months:.0f} months")

        # --- Load full image ---
        full_img_path = DATASET_DIR / "Tam_Görüntüler" / f"{sample_id}.png"

        pixmap = QPixmap(str(full_img_path))
        print("pixmap.isNull():", pixmap.isNull(), "label size:", self.image_lbl.size())

        if pixmap.isNull():
            self.image_lbl.setText("Image could not be loaded (pixmap NULL)")
            self.image_lbl.setPixmap(QPixmap())
        else:
            pixmap = pixmap.scaled(
                self.image_lbl.width(),
                self.image_lbl.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_lbl.setPixmap(pixmap)
            self.image_lbl.setText("")

def main():
    app = QApplication(sys.argv)
    w = BoneAgeApp()
    w.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
