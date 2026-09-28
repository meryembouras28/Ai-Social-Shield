from typing import List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.video_analyzer import VideoAnalyzer
from utils.video_processor import VideoProcessor


router = APIRouter()

video_processor = VideoProcessor()
video_analyzer = VideoAnalyzer(video_processor)


class VideoRequest(BaseModel):
    videoUrl: str = Field(
        ...,
        min_length=1,
    )

    thumbnailUrl: Optional[str] = Field(
        default=None,
        min_length=1,
    )

    description: Optional[str] = None

    comments: Optional[
        Union[
            List[str],
            List[dict],
            str,
        ]
    ] = None


@router.post("/classify/video")
def classify_video(request: VideoRequest):
    work_dir = None

    try:
        download_result = video_processor.download_video(
            request.videoUrl
        )

        video_path = download_result["video_path"]
        work_dir = download_result["work_dir"]

        external_thumbnail_path = None

        if request.thumbnailUrl:
            external_thumbnail_path = (
                work_dir / "external_thumbnail.jpg"
            )

            try:
                video_processor.download_external_file(
                    request.thumbnailUrl,
                    external_thumbnail_path,
                )
            except Exception:
                external_thumbnail_path = None

        result = video_analyzer.analyze_video_file(
            video_path=video_path,
            thumbnail_path=external_thumbnail_path,
            description=request.description,
            comments=request.comments,
            subtitle_path=download_result.get(
                "subtitle_path"
            ),
        )

        result["videoUrl"] = request.videoUrl
        result["thumbnailUrl"] = request.thumbnailUrl
        result["title"] = download_result.get("title")
        result["duration"] = download_result.get("duration")

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant l'analyse vidéo: "
                f"{str(exc)}"
            ),
        )

    finally:
        if work_dir is not None:
            video_processor.cleanup(work_dir)
