# Use a slim version of Python
FROM python:3.9-slim

# 1. Install modern system dependencies for OpenCV and Dlib
# 'libgl1' and 'libmesa-glx' are the updated replacements
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the entire project
COPY . .

EXPOSE 8000

# 4. Run the API
CMD ["python", "-m", "app.main_api"]