FROM python:3.12-slim

# Configure Python for container use
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

# Set the project root inside the image
WORKDIR /app

# Install system dependencies (kept minimal for slim images)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default entrypoint runs the Tensor-DMD pipeline; override or append args when needed
ENTRYPOINT ["python", "DMD Tensor/DMD_Tensor.py"]
# By default, execute the full comparison and persist artefacts under DMD Tensor/artifacts
CMD ["--data-path", "textile_machine_data.csv", "--compare-classifiers", "--include-baseline", "--grid-search", "--output-dir", "DMD Tensor/artifacts"]
