from io import BytesIO
import logging

import requests
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel, HttpUrl

from models.vision_classifier import classify_image


router = APIRouter()
logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 20 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/tiff",
    "application/octet-stream",
}


class ImageRequest(BaseModel):
    imageUrl: HttpUrl


@router.post("/classify/image")
def classify_image_endpoint(request: ImageRequest):
    image_url = str(request.imageUrl)

    logger.info("Analyse image demandée : %s", image_url)

    try:
        response = requests.get(
            image_url,
            timeout=30,
            stream=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/139.0 Safari/537.36"
                )
            },
            allow_redirects=True,
        )
        response.raise_for_status()

    except requests.Timeout:
        logger.exception("Timeout lors du téléchargement : %s", image_url)
        raise HTTPException(
            status_code=408,
            detail="Timeout lors du téléchargement de l'image",
        )

    except requests.RequestException as e:
        logger.exception("Impossible de télécharger l'image : %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de télécharger l'image : {str(e)}",
        )

    content_type = (
        response.headers
        .get("Content-Type", "")
        .split(";")[0]
        .strip()
        .lower()
    )

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Format d'image non supporté.",
        )

    content_length = response.headers.get("Content-Length")

    if content_length:
        try:
            if int(content_length) > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="La taille maximale de l'image est de 20 MB",
                )
        except ValueError:
            pass

    try:
        image_bytes = response.content

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="La taille maximale de l'image est de 20 MB",
            )

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="L'image téléchargée est vide",
            )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Impossible de lire l'image : %s", e)
        raise HTTPException(
            status_code=400,
            detail="Impossible de lire l'image téléchargée",
        )

    try:
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream)
        image.verify()

        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream).convert("RGB")

    except Exception as e:
        logger.exception("Image invalide : %s", e)
        raise HTTPException(
            status_code=400,
            detail="Le fichier téléchargé n'est pas une image valide",
        )

    try:
        result = classify_image(image)

    except Exception as e:
        logger.exception("Erreur pendant l'analyse Vision IA : %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne pendant l'analyse de l'image : {str(e)}",
        )

    scores = result.get("scores", {})

    if not isinstance(scores, dict):
        scores = {}

    response_data = {
        "imageUrl": image_url,
        "scores": {
            "nudite": round(float(scores.get("nudite", 0.0)), 2),
            "violence": round(float(scores.get("violence", 0.0)), 2),
            "armes": round(float(scores.get("armes", 0.0)), 2),
            "sang": round(float(scores.get("sang", 0.0)), 2),
            "logos_pub": round(float(scores.get("logos_pub", 0.0)), 2),
            "contenu_choquant": round(
                float(scores.get("contenu_choquant", 0.0)), 2
            ),
        },
        "ocrText": result.get("ocrText") or None,
    }

    logger.info("Analyse image terminée : %s", image_url)

    return response_data