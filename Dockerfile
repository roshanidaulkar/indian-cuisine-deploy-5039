# Use an official Python runtime as a parent image
FROM public.ecr.aws/docker/library/python:3.10-slim

# Update pip first
RUN pip install --no-cache-dir --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Use a build argument to pass AWS credentials
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Set AWS credentials as environment variables
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install awscli

# Download the CSV dataset from S3
RUN aws s3 cp s3://data-repo-cleaned-us-east-1-5030/ProcessedCleaned.csv /app/ProcessedCleaned.csv

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
