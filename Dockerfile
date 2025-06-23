# Use an official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements files (adjust if using poetry/pipenv)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code to the container
COPY . .

# Expose the port used by FastAPI/Uvicorn
EXPOSE 8002

# Command to run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]