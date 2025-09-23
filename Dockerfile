FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project configuration first for better caching
COPY app/bike_project/pyproject.toml ./bike_project/

# Install dependencies
RUN pip install --upgrade pip && pip install -e bike_project[dev]

# Copy the entire bike_project
COPY app/bike_project ./bike_project

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=", "--NotebookApp.password="]