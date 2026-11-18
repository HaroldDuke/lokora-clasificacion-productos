"""Servicio propuesto; requiere un run entrenado y revisado.

LOKORA_MODEL_DIR=/ruta/al/run uvicorn servicio_inferencia:app --port 8000
No modifica la API de pagos ni publica el modelo automáticamente.
"""
from contextlib import asynccontextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile

MAX_BYTES = 5 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000


@asynccontextmanager
async def lifespan(app):
    configured = os.environ.get('LOKORA_MODEL_DIR')
    if not configured:
        raise RuntimeError('Defina LOKORA_MODEL_DIR con un run entrenado y revisado.')
    directory = Path(configured)
    app.state.model = tf.keras.models.load_model(directory/'modelo.keras')
    app.state.classes = json.loads((directory/'clases.json').read_text())
    app.state.metadata = json.loads((directory/'metadatos.json').read_text())
    yield


app = FastAPI(title='Lokora sugerencia de categoría', version='0.1', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status':'ok', 'model_version':app.state.metadata['run_id']}


@app.post('/predict-category')
async def predict_category(image: UploadFile = File(...)):
    raw = await image.read(MAX_BYTES+1)
    if not raw or len(raw) > MAX_BYTES:
        raise HTTPException(413, 'La imagen debe ocupar entre 1 byte y 5 MB.')
    try:
        with Image.open(BytesIO(raw)) as im:
            if im.width * im.height > 25_000_000:
                raise HTTPException(413, 'La imagen excede 25 megapíxeles.')
            array = np.asarray(ImageOps.exif_transpose(im).convert('RGB'),dtype=np.float32)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, 'No se pudo decodificar la imagen.') from exc
    tensor = tf.image.resize_with_pad(array,224,224)[None,...]
    started=time.perf_counter()
    scores=app.state.model(tensor,training=False).numpy()[0]
    elapsed=(time.perf_counter()-started)*1000
    top=int(np.argmax(scores))
    return {
        'suggested_category':app.state.classes[top],
        'scores':{c:float(s) for c,s in zip(app.state.classes,scores)},
        'max_softmax':float(scores[top]),
        'model_version':app.state.metadata['run_id'],
        'inference_ms':elapsed,
        'requires_human_confirmation':True,
        'scope':'Solo envasados frutas y verduras. La salida no detecta de forma fiable productos fuera del piloto.'
    }
