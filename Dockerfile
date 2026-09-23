FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

#Application Files
COPY app.py .
COPY cache.py .
COPY config.py .
COPY llm.py .
COPY memory.py .
COPY retrieval.py .
COPY bm25.json .

#Start Streamlit
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.fileWatcherType=none"]