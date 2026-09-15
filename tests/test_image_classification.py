from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


EXPECTED_CATEGORIES = {
    "nudite",
    "violence",
    "armes",
    "sang",
    "logos_pub",
    "contenu_choquant",
}


def test_image_jpeg():
    response = client.post(
        "/classify/image",
        json={
            "imageUrl": "https://httpbin.org/image/jpeg"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "imageUrl" in data
    assert "scores" in data
    assert "ocrText" in data

    scores = data["scores"]

    assert set(scores.keys()) == EXPECTED_CATEGORIES

    for score in scores.values():
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100


def test_image_png():
    response = client.post(
        "/classify/image",
        json={
            "imageUrl": "https://httpbin.org/image/png"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "scores" in data

    scores = data["scores"]

    assert set(scores.keys()) == EXPECTED_CATEGORIES


def test_url_invalide():
    response = client.post(
        "/classify/image",
        json={
            "imageUrl": "ceci-nest-pas-une-url"
        },
    )

    assert response.status_code == 422


def test_image_url_manquante():
    response = client.post(
        "/classify/image",
        json={},
    )

    assert response.status_code == 422