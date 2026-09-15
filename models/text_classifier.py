from functools import lru_cache
import logging

from transformers import pipeline

from utils.config import MODEL_NAME


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_classifier():
    logger.info("Chargement du modèle Zero-Shot : %s", MODEL_NAME)

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
    )

    logger.info("Modèle Zero-Shot chargé avec succès.")

    return classifier