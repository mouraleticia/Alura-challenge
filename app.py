"""
app.py
------
Interface do Alura Agent em Streamlit. Simples e funcional: campo de chat,
indicação clara de que é um agente de IA, histórico de conversa dentro da
sessão e exibição das fontes usadas em cada resposta.

Rodar localmente:
    streamlit run app.py
"""

import streamlit as st

from rag_chain import responder

st.set_page_config(page_title="Alura Agent", page_icon="🤖")

st.title("🤖 Alura Agent")
st.caption(
    "Você está conversando com um **agente de IA** que responde com base nos "
    "documentos internos indexados. Ele não tem conhecimento além desses documentos."
)

# Histórico de conversa em memória (dura enquanto a sessão do navegador estiver aberta).
if "historico" not in st.session_state:
    st.session_state.historico = []

# Reexibe as mensagens anteriores da sessão a cada rerun do Streamlit.
for mensagem in st.session_state.historico:
    with st.chat_message(mensagem["papel"]):
        st.markdown(mensagem["conteudo"])
        if mensagem.get("fontes"):
            with st.expander("📎 Fontes usadas nesta resposta"):
                for fonte in mensagem["fontes"]:
                    st.markdown(f"- {fonte}")

pergunta = st.chat_input("Digite sua pergunta sobre os documentos...")

if pergunta:
    # Mostra a pergunta do usuário e guarda no histórico.
    st.session_state.historico.append({"papel": "user", "conteudo": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Gera a resposta via pipeline RAG e mostra com indicador de carregamento.
    with st.chat_message("assistant"):
        with st.spinner("Buscando nos documentos..."):
            resultado = responder(pergunta)
        st.markdown(resultado["resposta"])

        if resultado["fontes"]:
            with st.expander("📎 Fontes usadas nesta resposta"):
                for fonte in resultado["fontes"]:
                    st.markdown(f"- {fonte}")

        # Botão de feedback simples (positivo/negativo) — não persiste ainda,
        # mas já cobre o requisito de UI. Pode ser conectado a um log depois.
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("👍", key=f"like_{len(st.session_state.historico)}")
        with col2:
            st.button("👎", key=f"dislike_{len(st.session_state.historico)}")

    st.session_state.historico.append(
        {
            "papel": "assistant",
            "conteudo": resultado["resposta"],
            "fontes": resultado["fontes"],
        }
    )