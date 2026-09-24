FROM node:22-bookworm-slim AS mediapipe-assets

WORKDIR /assets
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
COPY startup_qa.py .
COPY blink_schedule.py .
COPY gaze_schedule.py .
COPY image_normalization.py .
COPY natural_motion.py .
COPY optional_services.py .
COPY speech_contract.py .
COPY static ./static
COPY --from=mediapipe-assets /assets/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs ./static/vendor/mediapipe/vision_bundle.mjs
COPY --from=mediapipe-assets /assets/node_modules/@mediapipe/tasks-vision/wasm ./static/vendor/mediapipe/wasm

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
