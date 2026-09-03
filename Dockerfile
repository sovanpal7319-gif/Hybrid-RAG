FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only
# rebuilds (torch/transformers are large and slow to reinstall otherwise).
COPY requirements.txt .
# torch is pulled from the CPU-only wheel index first: the default PyPI
# wheel for Linux drags in several GB of separate nvidia-*/cuda-* packages
# for GPU support this service never uses (no GPU in this deployment), which
# is pure download weight and a bigger surface for a flaky-connection failure.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
