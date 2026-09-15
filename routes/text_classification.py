from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from models.text_classifier import get_classifier


logger = logging.getLogger(__name__)

router = APIRouter()


CATEGORIES = [
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


CATEGORY_KEYS = {
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


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )


@router.post("/classify/text")
def classify_text(request: TextRequest):
    logger.info(
        "Classification demandée - longueur du texte : %d caractères",
        len(request.text),
    )

    try:
        classifier = get_classifier()

    except Exception:
        logger.exception("Modèle indisponible")
        raise HTTPException(
            status_code=503,
            detail="Modèle IA indisponible, réessayez plus tard",
        )

    try:
        result = classifier(
            request.text,
            candidate_labels=CATEGORIES,
            multi_label=True,
        )

    except Exception:
        logger.exception("Erreur pendant la classification")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant la classification",
        )

    categories = {}

    for label, score in zip(
        result["labels"],
        result["scores"],
    ):
        category_key = CATEGORY_KEYS.get(label)

        if category_key is not None:
            categories[category_key] = round(
                float(score) * 100,
                2,
            )

    for category_key in CATEGORY_KEYS.values():
        categories.setdefault(category_key, 0.0)

    response = {
        "text": request.text,
        "categories": categories,
    }

    logger.info("Classification terminée avec succès")

    return response