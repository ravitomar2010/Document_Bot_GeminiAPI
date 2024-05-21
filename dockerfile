FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY geminichat.py .
COPY OIP.jpeg .

ENTRYPOINT ["streamlit", "run", "geminichat.py"]

