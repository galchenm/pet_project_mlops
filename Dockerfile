# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and models into the container
COPY src ./src
COPY models ./models

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "src.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
