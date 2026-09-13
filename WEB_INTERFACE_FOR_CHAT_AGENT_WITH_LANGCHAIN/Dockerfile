# Image base
FROM python:3.12

# Dependencies installation
RUN apt-get update && apt-get install -y build-essential

# Define working directory
WORKDIR /app

# Copy requirements file from local to container working directory
COPY requirements.txt .

# Add local content to container working directory
ADD . /app

# Execute packages instalation in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#Expose port to access app
EXPOSE 8501

# Execute app
CMD ["streamlit", "run", "app.py"]