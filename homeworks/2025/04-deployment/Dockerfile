FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN apt-get update && apt-get install -y build-essential

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync

# Copy the prediction script
COPY ./src/main_docker.py ./

# Set the entrypoint
ENTRYPOINT ["uv", "run", "main_docker.py"]