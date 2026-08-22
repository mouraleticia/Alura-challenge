# Imagem base leve com Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema necessárias para pypdf/chromadb/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências Python primeiro (aproveita cache do Docker)
COPY requirements.txt .

# Instala o PyTorch na versão CPU-only ANTES do resto — evita que o
# sentence-transformers puxe a versão com suporte a GPU CUDA (muito maior
# e inútil aqui, já que a instância não tem GPU nenhuma).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação
COPY . .

# Porta padrão do Streamlit
EXPOSE 8501

# Healthcheck simples para orquestradores/monitoramento
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicialização — expõe em 0.0.0.0 para ser acessível fora do container
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]