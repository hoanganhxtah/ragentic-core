FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY app/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY app/ ./app/

RUN addgroup --system agent && \
    adduser --system --ingroup agent agent && \
    chown -R agent:agent /app

USER agent

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
