FROM python:3.12-slim


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src ./src
COPY models ./models


EXPOSE 8000


CMD ["uvicorn", "src.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
