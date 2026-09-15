from pathlib import Path
import re

import cv2

from models.text_classifier import get_classifier
from models.vision_classifier import classify_image, extract_ocr
from models.video_transcriber import transcribe_audio


TEXT_CATEGORIES = [
    "publicité",
    "spam",
    "politique",
    "religion",
    "violence",
    "haine",
    "fake news",
    "clickbait",
    "crypto",
    "paris sportifs",
    "arnaques",
]


TEXT_CATEGORY_KEYS = {
    "publicité": "publicite",
    "spam": "spam",
    "politique": "politique",
    "religion": "religion",
    "violence": "violence",
    "haine": "haine",
    "fake news": "fake_news",
    "clickbait": "clickbait",
    "crypto": "crypto",
    "paris sportifs": "paris_sportifs",
    "arnaques": "arnaques",
}


IMAGE_CATEGORIES = [
    "nudite",
    "violence",
    "armes",
    "sang",
    "logos_pub",
    "contenu_choquant",
]


def normalize_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_ocr_from_frames(frame_paths):
    texts = []
    frame_results = []

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))

        if image is None:
            frame_results.append({"text": ""})
            continue

        try:
            text = extract_ocr(image)
        except Exception as exc:
            frame_results.append({
                "text": "",
                "error": str(exc),
            })
            continue

        text = normalize_text(text)

        frame_results.append({
            "text": text,
        })

        if text:
            texts.append(text)

    unique_texts = []
    seen = set()

    for text in texts:
        normalized = normalize_text(text)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        unique_texts.append(text)

    return {
        "text": " ".join(unique_texts).strip(),
        "frames": frame_results,
    }


def classify_video_text(text):
    empty_scores = {
        key: 0.0
        for key in TEXT_CATEGORY_KEYS.values()
    }

    if not text or not text.strip():
        return {"scores": empty_scores}

    classifier = get_classifier()

    result = classifier(
        text,
        candidate_labels=TEXT_CATEGORIES,
        multi_label=True,
    )

    scores = {
        TEXT_CATEGORY_KEYS[label]: round(float(score) * 100, 2)
        for label, score in zip(
            result["labels"],
            result["scores"],
        )
        if label in TEXT_CATEGORY_KEYS
    }

    for key in TEXT_CATEGORY_KEYS.values():
        scores.setdefault(key, 0.0)

    return {"scores": scores}


def fuse_scores(image_scores, text_scores):
    all_categories = set(image_scores) | set(text_scores)

    final_scores = {}

    for category in all_categories:
        image_score = float(image_scores.get(category, 0.0))
        text_score = float(text_scores.get(category, 0.0))

        final_scores[category] = round(
            0.30 * image_score + 0.70 * text_score,
            2,
        )

    return final_scores


class VideoAnalyzer:

    def __init__(self, processor):
        self.processor = processor

    def analyze_video_file(self, video_path):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Vidéo introuvable : {video_path}"
            )

        thumbnail_path = self.processor.extract_thumbnail(
            video_path
        )

        frame_paths = self.processor.extract_key_frames(
            video_path,
            frame_count=5,
        )

        audio_path = self.processor.extract_audio(
            video_path
        )

        transcription = transcribe_audio(audio_path)
        transcript_text = transcription.get("text", "")

        ocr_result = extract_ocr_from_frames(frame_paths)
        ocr_text = ocr_result["text"]

        combined_text = " ".join(
            part
            for part in [transcript_text, ocr_text]
            if part
        ).strip()

        text_scores = classify_video_text(
            combined_text
        )["scores"]

        images_to_analyze = [str(thumbnail_path)]

        for frame_path in frame_paths:
            frame_path = str(frame_path)

            if frame_path not in images_to_analyze:
                images_to_analyze.append(frame_path)

        image_scores = {
            category: 0.0
            for category in IMAGE_CATEGORIES
        }

        for image_path in images_to_analyze:
            try:
                result = classify_image(image_path)
            except Exception:
                continue

            if not isinstance(result, dict):
                continue

            scores = result.get("scores", {})

            if not isinstance(scores, dict):
                continue

            for category in IMAGE_CATEGORIES:
                score = float(
                    scores.get(category, 0.0)
                )

                image_scores[category] = max(
                    image_scores[category],
                    score,
                )

        image_scores = {
            category: round(score, 2)
            for category, score in image_scores.items()
        }

        final_scores = fuse_scores(
            image_scores,
            text_scores,
        )

        return {
            "transcription": transcription,
            "ocr": ocr_result,
            "combinedText": combined_text,
            "imageScores": image_scores,
            "textScores": text_scores,
            "finalScores": final_scores,
            "weights": {
                "image": 0.30,
                "text": 0.70,
            },
        }