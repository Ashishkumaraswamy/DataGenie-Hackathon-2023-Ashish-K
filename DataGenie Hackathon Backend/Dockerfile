# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    libgomp1
    
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "80"]
