FROM python:3.11-slim 
# Uses a lightweight Python 3.11 image as the base.

WORKDIR /app
# Creates and sets /app as the working directory inside the container.

COPY requirements.txt .
# Copies requirements.txt from the project folder into /app.

RUN pip install --no-cache-dir -r requirements.txt
# Installs all Python packages listed in requirements.txt without keeping pip's download cache.

COPY . .
# Copies the rest of the project files into /app.

EXPOSE 8000
# Documents that the application inside the container uses port 8000.

CMD ["uvicorn", "main:app", "--host","0.0.0.0", "--port", "8000"]
# Starts the FastAPI application using Uvicorn on port 8000 and all network interfaces.