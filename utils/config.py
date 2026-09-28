import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
)