from pathlib import Path

from models.vision_classifier import classify_image


IMAGE_PATH = Path("tests/assets/thumbnail.jpg")

EXPECTED_CATEGORIES = {
    "nudite",
    "violence",
    "armes",
    "sang",
    "logos_pub",
    "contenu_choquant",
}


def test_classify_image():
    assert IMAGE_PATH.exists(), (
        f"Image introuvable : {IMAGE_PATH}"
    )

    result = classify_image(IMAGE_PATH)

    assert isinstance(result, dict)

    assert "scores" in result
    assert "ocrText" in result

    scores = result["scores"]

    assert isinstance(scores, dict)
    assert set(scores.keys()) == EXPECTED_CATEGORIES

    for score in scores.values():
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

    assert isinstance(result["ocrText"], str)