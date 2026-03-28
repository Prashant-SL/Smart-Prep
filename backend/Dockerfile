# Use the official Python base image
FROM python:3.11-slim

# Set environment variables for the image
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Create and set the working directory
WORKDIR $APP_HOME

# Install dependencies needed for deployment (gunicorn, uvicorn, etc.)
# We copy requirements.txt and install it first to utilize Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the image
COPY . .

# Set the command to run the application (Gunicorn worker)
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker main:app