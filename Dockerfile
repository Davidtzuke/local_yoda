FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY yoda/ yoda/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["yoda", "web", "--host", "0.0.0.0"]
