# Use the official Python 3.11 slim image as the base image
FROM python:3.11-slim 
# Set the working directory in the container to /app
WORKDIR /app 

# Copy the requirements.txt file to the working directory in the container
COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt
COPY src/ src/

# Expose port 8000 for the application to be accessible from outside the container
EXPOSE 8000 

# Set the command to run the application using uvicorn when the container starts
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

