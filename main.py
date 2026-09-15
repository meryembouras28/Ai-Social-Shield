from fastapi import FastAPI

from routes.text_classification import router as text_classification_router
from routes.image_classification import router as image_router
from routes.video_classification import router as video_router


app = FastAPI(
    title="AI Social Shield",
)


app.include_router(text_classification_router)
app.include_router(image_router)
app.include_router(video_router)