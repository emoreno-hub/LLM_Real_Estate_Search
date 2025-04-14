# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all app files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "homematch_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
