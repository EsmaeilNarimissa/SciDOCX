# DeepSeek-OCR vLLM Docker Image
# Based on official vLLM OpenAI image for better compatibility
FROM vllm/vllm-openai:v0.8.5

# Switch to root user to install packages
USER root

# Set working directory
WORKDIR /app

# Copy the DeepSeek-OCR implementation and config
COPY DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-master/DeepSeek-OCR-vllm/ ./DeepSeek-OCR-vllm/

# Copy application files
COPY start_server.py .
COPY requirements.txt .

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    PyMuPDF \
    img2pdf \
    einops \
    easydict \
    addict \
    Pillow \
    numpy

# Add the DeepSeek-OCR directory to PYTHONPATH
ENV PYTHONPATH="/app:/app/DeepSeek-OCR-vllm:${PYTHONPATH}"

# Create directories for models and outputs
RUN mkdir -p /app/models /app/outputs

# Set proper permissions
RUN chmod +x /app/start_server.py

# Expose the API port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/usr/bin/python3", "/app/start_server.py"]