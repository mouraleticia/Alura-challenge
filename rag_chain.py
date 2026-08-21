"""
rag_chain.py
------------
Responsável pela ETAPA 3 do pipeline: recuperação de contexto (retrieval),
geração de resposta com o Gemini, citação de fontes e fallback anti-alucinação.

Este módulo é importado pelo app.py (interface Streamlit), mas também pode
ser testado isoladamente:

    python rag_chain.py "qual é o benefício do nível Estrategista?"
"""

import os
import sys

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # carrega GOOGLE_API_KEY do arquivo .env

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Modelo Gemini usado para gerar as respostas — "flash" tem camada gratuita
# generosa (sem cartão de crédito). Pode ser sobrescrito via variável de
# ambiente GOOGLE_MODEL sem precisar mexer no código.
MODELO_GEMINI = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")


# Quantos chunks recuperar por pergunta.
TOP_K = 4

# Limiar de distância abaixo do qual consideramos que NÃO há contexto
# relevante o suficiente (Chroma usa distância: quanto MENOR, mais similar).
# Ajuste esse valor testando com suas próprias perguntas.
LIMIAR_DISTANCIA = 1.0

PROMPT_SISTEMA = """Você é o Alura Agent, um assistente que responde perguntas
EXCLUSIVAMENTE com base no contexto fornecido abaixo, extraído de documentos
internos da empresa.

Regras obrigatórias:
1. Use apenas as informações do CONTEXTO. Nunca use conhecimento externo.
2. Se o contexto não contiver a informação necessária para responder,
   diga claramente: "Não encontrei essa informação nos documentos disponíveis."
   Não tente adivinhar ou completar a resposta com suposições.
3. Seja direto e claro na resposta, em português.
4. Não é necessário repetir as fontes no texto da resposta — elas serão
   exibidas separadamente pela interface.

CONTEXTO:
{contexto}
"""


def _carregar_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name="alura_agent",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def _formatar_fonte(metadata: dict) -> str:
    """Monta uma string legível de citação a partir dos metadados do chunk."""
    arquivo = metadata.get("arquivo", "desconhecido")
    if "pagina" in metadata:
        return f"{arquivo} (página {metadata['pagina']})"
    if "linha" in metadata:
        return f"{arquivo} (linha {metadata['linha']})"
    return arquivo


def _extrair_texto(conteudo) -> str:
    """
    Extrai apenas o texto da resposta do LLM. Modelos mais novos do Gemini
    (3.x) retornam o conteúdo como uma lista de blocos (com metadados
    internos de verificação), em vez de uma string simples — esta função
    lida com os dois formatos.
    """
    if isinstance(conteudo, str):
        return conteudo

    if isinstance(conteudo, list):
        partes = []
        for bloco in conteudo:
            if isinstance(bloco, str):
                partes.append(bloco)
            elif isinstance(bloco, dict) and bloco.get("type") == "text":
                partes.append(bloco.get("text", ""))
        return "".join(partes).strip()

    return str(conteudo)


def responder(pergunta: str) -> dict:
    """
    Executa o pipeline RAG completo para uma pergunta:
    1. Busca os chunks mais relevantes no Chroma.
    2. Verifica se há contexto relevante o suficiente (fallback anti-alucinação).
    3. Monta o prompt e chama o Gemini.
    4. Retorna a resposta junto com a lista de fontes usadas.

    Retorno: {"resposta": str, "fontes": list[str]}
    """
    vectorstore = _carregar_vectorstore()

    resultados = vectorstore.similarity_search_with_score(pergunta, k=TOP_K)

    # Filtra apenas os chunks com distância abaixo do limiar (mais similares).
    resultados_relevantes = [
        (doc, score) for doc, score in resultados if score <= LIMIAR_DISTANCIA
    ]

    if not resultados_relevantes:
        return {
            "resposta": "Não encontrei essa informação nos documentos disponíveis.",
            "fontes": [],
        }

    contexto = "\n\n---\n\n".join(
        f"[Fonte: {_formatar_fonte(doc.metadata)}]\n{doc.page_content}"
        for doc, _ in resultados_relevantes
    )

    llm = ChatGoogleGenerativeAI(model=MODELO_GEMINI, temperature=0)

    mensagens = [
        ("system", PROMPT_SISTEMA.format(contexto=contexto)),
        ("human", pergunta),
    ]

    resposta = llm.invoke(mensagens)

    fontes = sorted(set(_formatar_fonte(doc.metadata) for doc, _ in resultados_relevantes))

    return {"resposta": _extrair_texto(resposta.content), "fontes": fontes}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python rag_chain.py 'sua pergunta aqui'")
        sys.exit(1)

    pergunta_teste = " ".join(sys.argv[1:])
    resultado = responder(pergunta_teste)

    print("\nResposta:", resultado["resposta"])
    print("\nFontes:", ", ".join(resultado["fontes"]) if resultado["fontes"] else "nenhuma")