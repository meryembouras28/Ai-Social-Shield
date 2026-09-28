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

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_comments(comments):
    if not comments:
        return ""

    if isinstance(comments, str):
        return normalize_text(comments)

    if isinstance(comments, dict):
        for key in ["text", "comment", "content"]:
            value = comments.get(key)

            if value:
                return normalize_text(value)

        return ""

    if isinstance(comments, list):
        texts = []

        for comment in comments:
            if isinstance(comment, str):
                text = comment
            elif isinstance(comment, dict):
                text = (
                    comment.get("text")
                    or comment.get("comment")
                    or comment.get("content")
                    or ""
                )
            else:
                text = ""

            text = normalize_text(text)

            if text:
                texts.append(text)

        return " ".join(texts).strip()

    return ""


def read_subtitle_file(subtitle_path):
    if subtitle_path is None:
        return ""

    subtitle_path = Path(subtitle_path)

    if not subtitle_path.exists():
        return ""

    try:
        content = subtitle_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""

    lines = []

    for line in content.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.upper() == "WEBVTT":
            continue

        if line.isdigit():
            continue

        if "-->" in line:
            continue

        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"\{[^}]+\}", " ", line)
        line = normalize_text(line)

        if line:
            lines.append(line)

    unique_lines = []
    previous = None

    for line in lines:
        if line != previous:
            unique_lines.append(line)
            previous = line

    return " ".join(unique_lines).strip()


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
            frame_results.append(
                {
                    "text": "",
                    "error": str(exc),
                }
            )
            continue

        text = normalize_text(text)

        frame_results.append({"text": text})

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

    def analyze_video_file(
        self,
        video_path,
        thumbnail_path=None,
        description=None,
        comments=None,
        subtitle_path=None,
    ):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Vidéo introuvable : {video_path}"
            )

        if thumbnail_path is not None:
            thumbnail_path = Path(thumbnail_path)

            if not thumbnail_path.exists():
                thumbnail_path = self.processor.extract_thumbnail(
                    video_path
                )
        else:
            thumbnail_path = self.processor.extract_thumbnail(
                video_path
            )

        frame_paths = self.processor.extract_key_frames(
            video_path,
            frame_count=5,
        )

        audio_path = self.processor.extract_audio(video_path)
        transcription = transcribe_audio(audio_path)

        transcript_text = normalize_text(
            transcription.get("text", "")
        )

        ocr_result = extract_ocr_from_frames(frame_paths)

        ocr_text = normalize_text(
            ocr_result.get("text", "")
        )

        subtitle_text = read_subtitle_file(subtitle_path)

        description_text = normalize_text(
            description or ""
        )

        comments_text = normalize_comments(comments)

        text_parts = [
            transcript_text,
            subtitle_text,
            ocr_text,
            description_text,
            comments_text,
        ]

        combined_text = " ".join(
            part
            for part in text_parts
            if part
        ).strip()

        text_scores = classify_video_text(
            combined_text
        )["scores"]

        images_to_analyze = []

        if thumbnail_path.exists():
            images_to_analyze.append(str(thumbnail_path))

        for frame_path in frame_paths:
            frame_path = str(frame_path)

            if frame_path not in images_to_analyze:
                images_to_analyze.append(frame_path)

        image_scores = {
            category: 0.0
            for category in IMAGE_CATEGORIES
        }

        image_results = []

        for image_path in images_to_analyze:
            try:
                result = classify_image(image_path)
            except Exception as exc:
                image_results.append(
                    {
                        "image": image_path,
                        "error": str(exc),
                    }
                )
                continue

            if not isinstance(result, dict):
                continue

            scores = result.get("scores", {})

            if not isinstance(scores, dict):
                continue

            image_results.append(
                {
                    "image": image_path,
                    "scores": scores,
                    "ocrText": result.get("ocrText"),
                }
            )

            for category in IMAGE_CATEGORIES:
                score = float(scores.get(category, 0.0))

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
            "subtitles": {
                "text": subtitle_text,
                "source": (
                    str(subtitle_path)
                    if subtitle_path is not None
                    else None
                ),
            },
            "description": description_text,
            "comments": comments_text,
            "combinedText": combined_text,
            "imageScores": image_scores,
            "imageResults": image_results,
            "textScores": text_scores,
            "finalScores": final_scores,
            "weights": {
                "image": 0.30,
                "text": 0.70,
            },
        }
