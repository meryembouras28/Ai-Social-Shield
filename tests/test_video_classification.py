from pathlib import Path

from models.video_analyzer import VideoAnalyzer
from utils.video_processor import VideoProcessor


VIDEO_PATH = Path("tests/assets/test_video.mp4")


def test_video_analysis_local():
    assert VIDEO_PATH.exists()

    processor = VideoProcessor()
    analyzer = VideoAnalyzer(processor)

    result = analyzer.analyze_video_file(VIDEO_PATH)

    expected_keys = {
        "transcription",
        "ocr",
        "combinedText",
        "imageScores",
        "textScores",
        "finalScores",
        "weights",
    }

    assert expected_keys.issubset(result.keys())

    assert isinstance(result["transcription"], dict)
    assert isinstance(result["ocr"], dict)
    assert isinstance(result["combinedText"], str)

    assert isinstance(result["imageScores"], dict)
    assert isinstance(result["textScores"], dict)
    assert isinstance(result["finalScores"], dict)

    assert len(result["finalScores"]) > 0

    assert result["weights"] == {
        "image": 0.30,
        "text": 0.70,
    }


def test_video_key_frames_and_thumbnail():
    assert VIDEO_PATH.exists()

    processor = VideoProcessor()

    thumbnail = processor.extract_thumbnail(VIDEO_PATH)

    frames = processor.extract_key_frames(
        VIDEO_PATH,
        frame_count=5,
    )

    assert thumbnail.exists()
    assert len(frames) == 5

    for frame in frames:
        assert frame.exists()