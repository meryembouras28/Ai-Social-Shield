from pathlib import Path

from models.vision_classifier import classify_image


IMAGES_DIR = Path("tests/assets/images")

EXPECTED_CATEGORIES = {
    "nudite",
    "violence",
    "armes",
    "sang",
    "logos_pub",
    "contenu_choquant",
}


def assert_valid_image_result(result):
    assert isinstance(result, dict)
    assert "scores" in result
    assert "ocrText" in result

    scores = result["scores"]

    assert isinstance(scores, dict)
    assert set(scores.keys()) == EXPECTED_CATEGORIES

    for score in scores.values():
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

    assert result["ocrText"] is None or isinstance(
        result["ocrText"],
        str,
    )


def test_image_publicite():
    image_path = (
        IMAGES_DIR / "publicite_evidente.png"
    )

    assert image_path.exists(), (
        f"Image introuvable : {image_path}"
    )

    result = classify_image(image_path)

    assert_valid_image_result(result)


def test_image_cas_borderline():
    image_path = (
        IMAGES_DIR / "cas_borderline.png"
    )

    assert image_path.exists(), (
        f"Image introuvable : {image_path}"
    )

    result = classify_image(image_path)

    assert_valid_image_result(result)


def test_image_texte_neutre():
    image_path = (
        IMAGES_DIR / "texte_neutre.png"
    )

    assert image_path.exists(), (
        f"Image introuvable : {image_path}"
    )

    result = classify_image(image_path)

    assert_valid_image_result(result)

    assert result["ocrText"] is not None
    assert "PROFITER" in result["ocrText"].upper()