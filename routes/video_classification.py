from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.video_processor import VideoProcessor
from models.video_analyzer import VideoAnalyzer


router = APIRouter()

video_processor = VideoProcessor()
video_analyzer = VideoAnalyzer(video_processor)


class VideoRequest(BaseModel):
    videoUrl: str = Field(..., min_length=1)


@router.post("/classify/video")
def classify_video(request: VideoRequest):
    work_dir = None

    try:
        # ========================================================
        # 1. Télécharger la vidéo
        # ========================================================

        download_result = video_processor.download_video(
            request.videoUrl
        )

        video_path = download_result["video_path"]
        work_dir = download_result["work_dir"]

        # ========================================================
        # 2. Analyse Lot C
        # ========================================================

        result = video_analyzer.analyze_video_file(video_path)

        # ========================================================
        # 3. Informations générales
        # ========================================================

        result["videoUrl"] = request.videoUrl
        result["title"] = download_result.get("title")
        result["duration"] = download_result.get("duration")

        # ========================================================
        # 4. Retour API
        # ========================================================

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant l'analyse vidéo : {str(e)}"
        )

    finally:
        # ========================================================
        # Nettoyage des fichiers temporaires
        # ========================================================

        if work_dir is not None:
            video_processor.cleanup(work_dir)