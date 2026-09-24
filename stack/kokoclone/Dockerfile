FROM python:3.12-slim

ARG KOKOCLONE_REVISION=dd6bd3acd3010dc223978839761db74957195f98
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface TORCH_HOME=/models/torch \
    KOKOCLONE_ENABLED=1 PYTHONPATH=/opt/kokoclone

RUN apt-get update \
    && apt-get install -y --no-install-recommends git espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kokoclone
RUN git clone https://github.com/Ashish-Patnaik/kokoclone.git . \
    && git checkout "${KOKOCLONE_REVISION}"
COPY requirements.txt /tmp/requirements.txt
# KokoClone only needs CPU PyTorch in this service.  Installing the default
# wheel pulls the full CUDA 12 stack into an otherwise slim image.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.6.0 torchaudio==2.6.0 \
    && sed -i '/^torch==/d;/^torchaudio==/d' /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt
COPY app.py /opt/kokoclone_service.py
WORKDIR /models/kokoclone

EXPOSE 8098
CMD ["uvicorn", "kokoclone_service:app", "--app-dir", "/opt", "--host", "0.0.0.0", "--port", "8098"]
