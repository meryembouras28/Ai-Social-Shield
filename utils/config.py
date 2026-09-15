import os

from dotenv import load_dotenv


load_dotenv()


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
)