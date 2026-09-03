FROM python:3.12-slim

WORKDIR /app

# Kept separate from the backend's requirements.txt/Dockerfile: the frontend
# only needs Flask + requests, not the backend's ML stack (torch,
# transformers, sentence-transformers), so this image builds fast and stays
# small instead of dragging in gigabytes of unused dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ ./templates/

EXPOSE 5000

CMD ["python", "app.py"]
