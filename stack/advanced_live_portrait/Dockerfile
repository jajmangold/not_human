ARG MUSE_TALK_IMAGE=musetalk:v100
FROM ${MUSE_TALK_IMAGE}

ARG ALP_REPOSITORY=https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git
ARG ALP_REVISION=3bba732915e22f18af0d221b9c5c282990181f1b

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    dill==0.3.8 \
    python-multipart==0.0.32 \
    timm==1.0.9 \
    ultralytics==8.2.0

RUN git init /opt/advanced-live-portrait \
    && git -C /opt/advanced-live-portrait remote add origin "${ALP_REPOSITORY}" \
    && git -C /opt/advanced-live-portrait fetch --depth 1 origin "${ALP_REVISION}" \
    && git -C /opt/advanced-live-portrait checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/advanced-live-portrait rev-parse HEAD)" = "${ALP_REVISION}"

WORKDIR /app
COPY app.py .
COPY face_validation.py .

ENTRYPOINT ["python3.10", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8093"]
