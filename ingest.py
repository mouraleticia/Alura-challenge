"""
ingest.py
----------
Responsável pela ETAPA 1 e 2 do pipeline: leitura dos documentos (PDF/CSV),
limpeza do texto, chunking e indexação vetorial no Chroma.

Uso:
    python ingest.py --file regras_gamificacao.csv
    python ingest.py --file meu_documento.pdf

Depois de rodar, uma pasta ./chroma_db é criada (ou atualizada) com o
índice vetorial pronto para ser consultado pelo rag_chain.py.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Pasta onde o banco vetorial Chroma é persistido em disco.
CHROMA_DIR = "./chroma_db"

# Modelo de embeddings local e gratuito (não depende de API paga).
# Roda via sentence-transformers/HuggingFace na própria máquina/VM.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def limpar_texto(texto: str) -> str:
    """
    Remove ruídos comuns de extração: espaços duplicados, quebras de linha
    excessivas e caracteres de controle residuais.
    """
    texto = re.sub(r"\s+", " ", texto)  # colapsa espaços/quebras de linha múltiplas
    texto = texto.strip()
    return texto


def carregar_pdf(caminho: str) -> list[Document]:
    """
    Carrega um PDF nativo (texto digital, não escaneado) usando PyPDFLoader.
    Cada página vira um Document com metadado 'page' já preenchido.
    """
    loader = PyPDFLoader(caminho)
    paginas = loader.load()

    documentos = []
    nome_arquivo = Path(caminho).name
    for pagina in paginas:
        texto_limpo = limpar_texto(pagina.page_content)
        if not texto_limpo:
            continue  # pula páginas vazias (comum em PDFs escaneados sem OCR)
        documentos.append(
            Document(
                page_content=texto_limpo,
                metadata={
                    "arquivo": nome_arquivo,
                    "pagina": pagina.metadata.get("page", 0) + 1,  # 1-indexado p/ leitura humana
                },
            )
        )
    return documentos


def carregar_csv(caminho: str) -> list[Document]:
    """
    Carrega um CSV com pandas. Cada linha vira um Document, convertendo as
    colunas em um texto "coluna: valor" legível pelo LLM. Isso preserva o
    contexto de cada registro sem precisar dividir mais (chunking natural).
    """
    df = pd.read_csv(caminho)
    nome_arquivo = Path(caminho).name

    documentos = []
    for indice, linha in df.iterrows():
        partes = [f"{coluna}: {valor}" for coluna, valor in linha.items()]
        texto = limpar_texto(" | ".join(partes))
        documentos.append(
            Document(
                page_content=texto,
                metadata={
                    "arquivo": nome_arquivo,
                    "linha": indice + 2,  # +2 pois a linha 1 é o cabeçalho do CSV
                },
            )
        )
    return documentos


def dividir_em_chunks(documentos: list[Document]) -> list[Document]:
    """
    Divide documentos longos (tipicamente páginas de PDF) em chunks menores,
    com overlap para não cortar uma ideia no meio. Documentos já pequenos
    (como linhas de CSV) atravessam essa etapa praticamente intactos.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documentos)


def indexar(documentos: list[Document]) -> None:
    """
    Gera embeddings para cada chunk e salva no Chroma, persistindo em disco.
    Se o banco já existir, os novos documentos são adicionados a ele.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = Chroma(
        collection_name="alura_agent",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.add_documents(documentos)
    print(f"[OK] {len(documentos)} chunks indexados em '{CHROMA_DIR}'.")


def main():
    parser = argparse.ArgumentParser(description="Ingestão de documentos para o Alura Agent")
    parser.add_argument("--file", required=True, help="Caminho do arquivo PDF ou CSV a indexar")
    args = parser.parse_args()

    caminho = args.file
    extensao = Path(caminho).suffix.lower()

    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    if extensao == ".pdf":
        documentos_brutos = carregar_pdf(caminho)
    elif extensao == ".csv":
        documentos_brutos = carregar_csv(caminho)
    else:
        raise ValueError(f"Formato não suportado: {extensao} (use .pdf ou .csv)")

    print(f"[INFO] {len(documentos_brutos)} registros extraídos de '{caminho}'.")

    chunks = dividir_em_chunks(documentos_brutos)
    print(f"[INFO] {len(chunks)} chunks gerados após o splitting.")

    indexar(chunks)


if __name__ == "__main__":
    main()