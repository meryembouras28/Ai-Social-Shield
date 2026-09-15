from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_publicite_evidente():
    response = client.post(
        "/classify/text",
        json={
            "text": "PROMOTION ! Achetez maintenant et profitez de -70% !"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "categories" in data
    assert len(data["categories"]) == 11

    for score in data["categories"].values():
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100


def test_texte_neutre():
    response = client.post(
        "/classify/text",
        json={
            "text": (
                "Aujourd'hui, il fait beau. "
                "Je vais me promener avec mes amis."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "categories" in data
    assert len(data["categories"]) == 11


def test_cas_borderline():
    response = client.post(
        "/classify/text",
        json={
            "text": (
                "Découvrez cette nouvelle application "
                "qui pourrait vous intéresser."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "categories" in data
    assert len(data["categories"]) == 11


def test_texte_trop_long():
    long_text = "a" * 5001

    response = client.post(
        "/classify/text",
        json={"text": long_text},
    )

    assert response.status_code == 422


def test_texte_vide():
    response = client.post(
        "/classify/text",
        json={"text": ""},
    )

    assert response.status_code == 422