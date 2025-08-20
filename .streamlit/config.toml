FROM python:3.11-slim
WORKDIR /app

# System deps (optional, add if you need them later)
# RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8080 PYTHONUNBUFFERED=1 SHOW_DIAGNOSTICS=false
# IMPORTANT: your entry file is streamlit_app.py in repo root
CMD ["bash","-lc","streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT}"]
