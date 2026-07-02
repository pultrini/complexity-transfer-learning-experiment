# ============================================================
# Stage: mlflow — servidor de tracking (leve, baseado em pip)
# ============================================================
FROM python:3.14-slim AS mlflow

RUN pip install --no-cache-dir \
    mlflow==3.14.0 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.162

EXPOSE 5000


# ============================================================
# Stage: app — aplicação de treinamento (uv + GPU)
# ============================================================

FROM python:3.14-slim

WORKDIR /app

# Dependências de sistema mínimas (gcc para pacotes que compilam extensões nativas)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instala o uv (gerenciador de pacotes) direto do instalador oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copia apenas os arquivos de dependência primeiro, para aproveitar o
# cache de camadas do Docker — só reinstala pacotes se pyproject.toml
# ou uv.lock mudarem, não a cada alteração de código.
COPY pyproject.toml uv.lock ./

# --frozen: falha se o uv.lock estiver desatualizado em vez de regenerá-lo
#           silenciosamente (garante reprodutibilidade no servidor remoto).
# --no-dev: não instala o grupo "dev" (ex: ruff) na imagem de produção.
RUN uv sync --frozen --no-dev

# Agora copia o restante do código
COPY . .

ENV PATH="/app/.venv/bin:${PATH}"

# Ajuste para o entrypoint real do seu projeto
CMD ["uv", "run", "python", "-m", "src.main"]