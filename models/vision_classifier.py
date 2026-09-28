from functools import lru_cache
from pathlib import Path
import logging
import os

import cv2
import numpy as np
import pytesseract
import timm
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

load_dotenv(BASE_DIR / ".env")


SAFETY_MODEL_PATH = MODELS_DIR / "image-safety-classifier-s"
NUDITY_MODEL_PATH = MODELS_DIR / "nudity-detector"
VIOLENCE_MODEL_PATH = MODELS_DIR / "vit-base-violence-detection"
BLOOD_MODEL_PATH = MODELS_DIR / "bloodshotnet" / "yolo26s.onnx"
WEAPON_MODEL_PATH = MODELS_DIR / "weapon" / "model" / "best.pt"
LOGO_MODEL_PATH = MODELS_DIR / "logos" / "yolov8x_logo_best.pt"


logger = logging.getLogger(__name__)


TESSERACT_LANG = os.getenv(
    "TESSERACT_LANG",
    "fra+eng",
)

TESSERACT_CMD_PATH = os.getenv(
    "TESSERACT_CMD_PATH",
    "",
)


def configure_tesseract():
    candidates = []

    if TESSERACT_CMD_PATH:
        candidates.append(
            Path(TESSERACT_CMD_PATH)
        )

    candidates.extend(
        [
            Path(
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            ),
            Path(
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(
                candidate
            )

            logger.info(
                "Tesseract configuré : %s",
                candidate,
            )

            return True

    try:
        version = pytesseract.get_tesseract_version()

        logger.info(
            "Tesseract trouvé dans le PATH : %s",
            version,
        )

        return True

    except Exception:
        logger.warning(
            "Tesseract n'a pas pu être configuré."
        )

        return False


TESSERACT_AVAILABLE = configure_tesseract()


WEAPON_CLASSES = {
    "Blunt_Weapon",
    "Explosive",
    "Fire_Smoke",
    "Firearm",
    "Melee_Weapon",
}


def normalize_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack(
                [image, image, image],
                axis=-1,
            )

        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[:, :, :3]

        if image.dtype != np.uint8:
            image = np.clip(
                image,
                0,
                255,
            ).astype(np.uint8)

        return Image.fromarray(image).convert("RGB")

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    raise TypeError(
        f"Type d'image non supporté : {type(image)}"
    )


@lru_cache(maxsize=1)
def get_safety_classifier():
    if not SAFETY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle safety introuvable : {SAFETY_MODEL_PATH}"
        )

    logger.info(
        "Chargement du modèle safety : %s",
        SAFETY_MODEL_PATH,
    )

    classifier = pipeline(
        "image-classification",
        model=str(SAFETY_MODEL_PATH),
    )

    logger.info("Modèle safety chargé.")

    return classifier


def analyze_safety(image):
    classifier = get_safety_classifier()
    results = classifier(image)

    scores = {}

    for item in results:
        label = str(
            item["label"]
        ).strip().upper()

        scores[label] = float(
            item["score"]
        ) * 100.0

    return scores


@lru_cache(maxsize=1)
def get_nudity_classifier():
    if not NUDITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle nudité introuvable : {NUDITY_MODEL_PATH}"
        )

    logger.info(
        "Chargement du modèle nudité : %s",
        NUDITY_MODEL_PATH,
    )

    classifier = pipeline(
        "image-classification",
        model=str(NUDITY_MODEL_PATH),
    )

    logger.info("Modèle nudité chargé.")

    return classifier


def analyze_nudity(image):
    classifier = get_nudity_classifier()
    results = classifier(image)

    scores = {}

    for item in results:
        label = str(
            item["label"]
        ).strip().lower()

        scores[label] = float(
            item["score"]
        ) * 100.0

    return {
        "nudity_pornography": round(
            scores.get(
                "nudity_pornography",
                0.0,
            ),
            2,
        ),
        "safe_normal": round(
            scores.get(
                "safe_normal",
                0.0,
            ),
            2,
        ),
        "gore_bloodshed_violent": round(
            scores.get(
                "gore_bloodshed_violent",
                0.0,
            ),
            2,
        ),
    }


@lru_cache(maxsize=1)
def get_violence_model():
    if not VIOLENCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle violence introuvable : {VIOLENCE_MODEL_PATH}"
        )

    checkpoint_path = (
        VIOLENCE_MODEL_PATH / "pytorch_model.bin"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint violence introuvable : {checkpoint_path}"
        )

    logger.info(
        "Chargement du modèle violence avec timm..."
    )

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=2,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    logger.info("Modèle violence chargé.")

    return model


def prepare_violence_image(image):
    image = normalize_image(image)

    image = image.resize(
        (224, 224),
        Image.Resampling.BICUBIC,
    )

    image_array = np.asarray(
        image
    ).astype(np.float32)

    image_array /= 255.0

    mean = np.array(
        [0.5, 0.5, 0.5],
        dtype=np.float32,
    )

    std = np.array(
        [0.5, 0.5, 0.5],
        dtype=np.float32,
    )

    image_array = (
        image_array - mean
    ) / std

    return (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def analyze_violence(image):
    model = get_violence_model()

    pixel_values = prepare_violence_image(
        image
    )

    with torch.no_grad():
        logits = model(pixel_values)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    class_0 = float(
        probabilities[0]
    ) * 100.0

    class_1 = float(
        probabilities[1]
    ) * 100.0

    return {
        "class_0": round(class_0, 2),
        "class_1": round(class_1, 2),
    }


@lru_cache(maxsize=1)
def get_blood_classifier():
    if not BLOOD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle BloodshotNet introuvable : {BLOOD_MODEL_PATH}"
        )

    return YOLO(
        str(BLOOD_MODEL_PATH)
    )


def analyze_blood(image):
    model = get_blood_classifier()

    results = model(
        image,
        conf=0.25,
        verbose=False,
    )

    max_score = 0.0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(
                box.conf[0].item()
            )

            max_score = max(
                max_score,
                confidence * 100.0,
            )

    return round(
        max_score,
        2,
    )


@lru_cache(maxsize=1)
def get_weapon_model():
    if not WEAPON_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle armes introuvable : {WEAPON_MODEL_PATH}"
        )

    logger.info(
        "Chargement du modèle armes..."
    )

    model = YOLO(
        str(WEAPON_MODEL_PATH)
    )

    logger.info(
        "Modèle armes chargé."
    )

    return model


def analyze_weapons(image):
    model = get_weapon_model()

    results = model(
        image,
        conf=0.25,
        verbose=False,
    )

    detections = []
    max_score = 0.0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(
                box.conf[0].item()
            )

            class_id = int(
                box.cls[0].item()
            )

            class_name = model.names.get(
                class_id,
                str(class_id),
            )

            if class_name not in WEAPON_CLASSES:
                continue

            coordinates = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            score = confidence * 100.0

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(
                        score,
                        2,
                    ),
                    "bbox": coordinates,
                }
            )

            max_score = max(
                max_score,
                score,
            )

    return {
        "score": round(
            max_score,
            2,
        ),
        "detections": detections,
    }


@lru_cache(maxsize=1)
def get_logo_model():
    if not LOGO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle logos introuvable : {LOGO_MODEL_PATH}"
        )

    logger.info(
        "Chargement du modèle logos..."
    )

    model = YOLO(
        str(LOGO_MODEL_PATH)
    )

    logger.info(
        "Modèle logos chargé."
    )

    return model


def analyze_logos(image):
    model = get_logo_model()

    results = model(
        image,
        conf=0.25,
        verbose=False,
    )

    detections = []
    max_score = 0.0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(
                box.conf[0].item()
            )

            coordinates = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            score = confidence * 100.0

            detections.append(
                {
                    "class": "logo",
                    "confidence": round(
                        score,
                        2,
                    ),
                    "bbox": coordinates,
                }
            )

            max_score = max(
                max_score,
                score,
            )

    return {
        "score": round(
            max_score,
            2,
        ),
        "detections": detections,
    }
def _prepare_ocr_variants(image):
    image = normalize_image(image)

    rgb = np.array(image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    variants = [
        ("original", rgb),
        ("grayscale", gray),
    ]

    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    variants.append(
        ("upscaled", enlarged)
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    clahe_image = clahe.apply(
        enlarged
    )

    variants.append(
        ("clahe", clahe_image)
    )

    threshold = cv2.adaptiveThreshold(
        clahe_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    variants.append(
        ("adaptive_threshold", threshold)
    )

    return variants


def _ocr_single_variant(image, psm):
    try:
        data = pytesseract.image_to_data(
            image,
            lang=TESSERACT_LANG,
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )

    except Exception:
        logger.exception(
            "Erreur Tesseract avec PSM %s",
            psm,
        )

        return "", 0.0

    accepted_words = []
    confidences = []

    texts = data.get("text", [])
    confs = data.get("conf", [])

    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()

        if not text:
            continue

        try:
            confidence = float(
                confs[index]
            )

        except (
            ValueError,
            TypeError,
            IndexError,
        ):
            continue

        if confidence < 45:
            continue

        alnum_count = sum(
            char.isalnum()
            for char in text
        )

        if alnum_count == 0:
            continue

        alnum_ratio = (
            alnum_count / len(text)
        )

        if alnum_ratio < 0.45:
            continue

        accepted_words.append(text)
        confidences.append(confidence)

    if not accepted_words:
        return "", 0.0

    text = " ".join(
        accepted_words
    )

    mean_confidence = float(
        np.mean(confidences)
    )

    return text, mean_confidence


def extract_ocr(image):
    if not TESSERACT_AVAILABLE:
        logger.warning(
            "OCR ignoré : Tesseract indisponible."
        )

        return ""

    variants = _prepare_ocr_variants(
        image
    )

    psms = [6, 11, 12, 7]
    candidates = []

    for variant_name, variant_image in variants:
        for psm in psms:
            text, confidence = _ocr_single_variant(
                variant_image,
                psm,
            )

            if not text:
                continue

            alnum_chars = sum(
                char.isalnum()
                for char in text
            )

            total_chars = len(text)

            if total_chars == 0:
                continue

            alnum_ratio = (
                alnum_chars / total_chars
            )

            quality = (
                confidence * alnum_ratio
            )

            candidates.append(
                {
                    "variant": variant_name,
                    "psm": psm,
                    "text": text,
                    "confidence": confidence,
                    "quality": quality,
                }
            )

    if not candidates:
        logger.info(
            "Aucun texte OCR exploitable."
        )

        return ""

    best = max(
        candidates,
        key=lambda item: item["quality"],
    )

    logger.info(
        "OCR sélectionné : variant=%s, PSM=%s, "
        "confidence=%.2f, quality=%.2f",
        best["variant"],
        best["psm"],
        best["confidence"],
        best["quality"],
    )

    cleaned_words = []
    previous_normalized = None

    for word in best["text"].split():
        word = word.strip(
            ".,;:!?()[]{}<>\"'|\\/_=~`"
        )

        if not word:
            continue

        normalized = word.lower()

        if normalized == previous_normalized:
            continue

        if (
            len(normalized) == 1
            and not normalized.isalnum()
        ):
            continue

        cleaned_words.append(word)
        previous_normalized = normalized

    return " ".join(
        cleaned_words
    ).strip()


def classify_image(image):
    image = normalize_image(image)

    safety_scores = analyze_safety(image)

    contenu_choquant_score = safety_scores.get(
        "LABEL_0",
        0.0,
    )

    nudity_result = analyze_nudity(image)

    nudite_score = nudity_result[
        "nudity_pornography"
    ]

    violence_result = analyze_violence(image)

    violence_score = violence_result[
        "class_1"
    ]

    sang_score = analyze_blood(image)

    weapon_result = analyze_weapons(image)

    armes_score = weapon_result[
        "score"
    ]

    logo_result = analyze_logos(image)

    logos_pub_score = logo_result[
        "score"
    ]

    ocr_text = extract_ocr(image)

    scores = {
        "nudite": round(
            float(nudite_score),
            2,
        ),
        "violence": round(
            float(violence_score),
            2,
        ),
        "armes": round(
            float(armes_score),
            2,
        ),
        "sang": round(
            float(sang_score),
            2,
        ),
        "logos_pub": round(
            float(logos_pub_score),
            2,
        ),
        "contenu_choquant": round(
            float(contenu_choquant_score),
            2,
        ),
    }

    return {
        "scores": scores,
        "ocrText": ocr_text,
    }