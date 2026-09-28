# AI-Social-Shield

AI-Social-Shield est un projet d'analyse de contenu avec l'intelligence artificielle.

Il permet d'analyser du texte, des images et des vidéos afin d'obtenir des scores pour différentes catégories de contenu. Le projet utilise principalement du **NLP**, de la **Computer Vision**, de l'**OCR** et de la **transcription audio**.

## Ce que fait le projet

### Texte

Le texte est analysé avec un modèle Zero-Shot :

`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`

Les catégories utilisées sont :

* publicité
* spam
* politique
* religion
* violence
* haine
* fake news
* clickbait
* crypto
* paris sportifs
* arnaques

Chaque catégorie reçoit un score entre 0 et 100.

### Image

L'analyse d'une image utilise plusieurs modèles selon le type de contenu recherché :

* nudité
* violence
* armes
* sang
* logos / publicité
* contenu choquant

Le projet utilise également **Tesseract OCR** pour récupérer le texte présent dans les images.

Pour les logos, le projet utilise un modèle YOLO entraîné sur le dataset **LogoDet-3K**. Le modèle est utilisé pour détecter la présence de logos dans l'image.

### Vidéo

Pour une vidéo, plusieurs informations sont récupérées avant l'analyse :

* transcription audio avec Whisper
* 5 images clés
* OCR sur les images
* sous-titres lorsqu'ils sont disponibles
* description
* commentaires
* miniature de la vidéo

Le texte récupéré est analysé avec le modèle NLP et les images avec les modèles de vision.

Les deux parties sont ensuite combinées avec :

* **70 % pour l'analyse textuelle**
* **30 % pour l'analyse visuelle**

Par exemple, sur une vidéo publicitaire testée, la transcription a été reconnue correctement et le score `publicite` obtenu par l'analyse textuelle était de **97.99**. Après la combinaison texte + image, le score final `publicite` était de **68.59**.

Cette différence est normale : dans ce cas, `publicite` n'a pas reçu de score visuel, donc le score final correspond à **0.70 × 97.99 = 68.59**.

Les scores bruts par modalité (`imageScores` et `textScores`) sont retournés séparément dans la réponse.

## API

Le projet utilise **FastAPI**.

Les principales routes sont :

```text
POST /classify/text
POST /classify/image
POST /classify/video
```

La documentation Swagger est disponible ici :

```text
http://127.0.0.1:8000/docs
```

## Structure

```text
AI-Social-Shield/
│
├── models/
│   ├── text_classifier.py
│   ├── vision_classifier.py
│   ├── video_analyzer.py
│   └── video_transcriber.py
│
├── routes/
│   ├── text_classification.py
│   ├── image_classification.py
│   └── video_classification.py
│
├── utils/
│   ├── config.py
│   └── video_processor.py
│
├── tests/
│   └── assets/
│
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

```bash
git clone https://github.com/meryembouras28/AI-Social-Shield.git
cd AI-Social-Shield
```

Récupérer les poids des modèles avec Git LFS :

```bash
git lfs install
git lfs pull
```

Cette étape est nécessaire pour récupérer les vrais fichiers des modèles utilisés par le projet.

Créer l'environnement virtuel :

```bash
python -m venv venv
```

Sous Windows :

```powershell
.\venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Créer ensuite le fichier `.env` à partir de `.env.example`.

Le projet nécessite également **FFmpeg** et **Tesseract OCR**.

Sous Windows, si Tesseract n'est pas disponible dans le PATH, renseigner son chemin dans `.env` :

```text
TESSERACT_CMD_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Lancer l'API

```bash
uvicorn main:app --reload
```

Puis ouvrir :

```text
http://127.0.0.1:8000/docs
```

## Tests

```bash
python -m pytest tests/ -v
```

Les tests couvrent actuellement le texte, les images et les vidéos.

## À propos du projet

Cette partie du projet s'occupe de **l'analyse du contenu et de la production des scores**.

La décision finale de modération est séparée de cette analyse.
