FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /app
ENTRYPOINT [ "streamlit", "run" ]
CMD ["--server.address", "0.0.0.0", "--server.port", "8501", "app.py"]

