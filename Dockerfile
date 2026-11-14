FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt requirements-llm.txt ./
RUN pip install -r requirements.txt -r requirements-llm.txt

COPY src ./src
COPY data ./data

RUN useradd --create-home appuser
USER appuser

# Las credenciales (WHATSAPP_*, ANTHROPIC_API_KEY) se pasan en tiempo de ejecución; nunca van en la imagen.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
