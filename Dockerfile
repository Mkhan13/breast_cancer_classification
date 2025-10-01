FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update 
RUN apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
RUN rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the Cloud Run port
EXPOSE 8080
ENV PORT=8080

# Run Streamlit
CMD ["sh", "-c", "streamlit run main.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS false --server.headless true"]
