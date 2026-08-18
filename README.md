# 🤖 Alura Agent
 
Agente de inteligência artificial que responde perguntas em linguagem natural
sobre documentos internos da empresa (PDF ou CSV), usando RAG (Retrieval-Augmented
Generation) com **Gemini (Google)** e **Chroma** como banco vetorial.
 
Projeto desenvolvido como desafio final do curso de IA da Alura.
 
> Documento indexado neste projeto: `regras_gamificacao.csv` — regras do
> programa de fidelidade/gamificação (níveis, faixas de XP e benefícios) do
> case fictício **Mercado Central 24h**.
 
---
 
## 🏗️ Arquitetura
 
```
[PDF/CSV] → Extração (pypdf/pandas) → Limpeza → Chunking (LangChain)
                                                          ↓
                                          Embeddings (HuggingFace, local)
                                                          ↓
                                              Chroma (banco vetorial)
 
[Pergunta do usuário] → Embedding da pergunta → Busca por similaridade (top-k)
                                                          ↓
                                    Chunks relevantes + pergunta → Prompt
                                                          ↓
                                              Gemini (Google) → Resposta
                                                          ↓
                                    Resposta + Fontes citadas → Streamlit
```
 
**Componentes principais:**
 
| Camada | Tecnologia |
|---|---|
| Linguagem | Python |
| Orquestração do agente | LangChain |
| LLM (geração de resposta) | Gemini (Google) via `langchain-google-genai` — camada gratuita, sem cartão |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, gratuito) |
| Banco vetorial | Chroma (persistido em disco) |
| Interface | Streamlit |
| Deploy | Oracle Cloud Infrastructure (OCI Compute) — dentro do *Always Free* |
 
**Anti-alucinação:** o agente só responde com base nos trechos recuperados do
banco vetorial. Se nenhum trecho relevante for encontrado (abaixo de um limiar
de similaridade), ele responde explicitamente que não encontrou a informação,
em vez de inventar uma resposta.
 
**Citação de fontes:** cada resposta é acompanhada dos metadados de origem
(nome do arquivo + página, no caso de PDF, ou número da linha, no caso de CSV).
 
---
 
## 💬 Exemplos de perguntas e respostas
 
> Baseado no documento `regras_gamificacao.csv` indexado neste projeto.
 
**Pergunta:** Quais benefícios o nível Estrategista desbloqueia?
**Resposta:** Ofertas antecipadas no celular e uso do caixa rápido na loja física.
**Fonte:** `regras_gamificacao.csv (linha 3)`
 
**Pergunta:** Como faço para chegar ao nível Lenda Central?
**Resposta:** É preciso acumular mais de 5000 pontos de XP, mantendo compras
semanais consistentes e completando os desafios do bairro.
**Fonte:** `regras_gamificacao.csv (linha 5)`
 
**Pergunta:** Qual é a política de reembolso da empresa?
**Resposta:** Não encontrei essa informação nos documentos disponíveis.
**Fonte:** _(nenhuma — fora do escopo do documento indexado)_
 
---
 
## 🚀 Como rodar localmente
 
### 1. Pré-requisitos
- Python 3.11+
- Uma chave gratuita da API do Gemini, gerada em [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
### 2. Clonar o repositório e instalar dependências
 
```bash
git clone https://github.com/SEU_USUARIO/alura-agent.git
cd alura-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
### 3. Configurar variáveis de ambiente
 
```bash
cp .env.example .env
# edite o .env e insira sua GOOGLE_API_KEY
```
 
### 4. Indexar o documento
 
```bash
python ingest.py --file regras_gamificacao.csv
```
 
### 5. Rodar a interface
 
```bash
streamlit run app.py
```
 
Acesse `http://localhost:8501` no navegador.
 
---
 
## ☁️ Deploy na Oracle Cloud (OCI Compute)
 
### 1. Criar a instância
No console da OCI: **Compute → Instances → Create Instance**, escolha uma
shape (ex: `VM.Standard.E2.1.Micro`, elegível no free tier), imagem **Ubuntu
22.04**, e associe um IP público.
 
### 2. Liberar a porta 8501
Em **Networking → Virtual Cloud Networks → sua VCN → Security Lists**,
adicione uma *Ingress Rule*: origem `0.0.0.0/0`, protocolo TCP, porta `8501`.
 
Dentro da própria instância, libere também no firewall do Ubuntu (se ativo):
 
```bash
sudo ufw allow 8501/tcp
```
 
### 3. Conectar via SSH e instalar o Docker
 
```bash
ssh ubuntu@SEU_IP_PUBLICO
 
sudo apt update && sudo apt install -y docker.io git
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# saia e reconecte via SSH para aplicar o grupo docker
```
 
### 4. Clonar o projeto e configurar o .env
 
```bash
git clone https://github.com/SEU_USUARIO/alura-agent.git
cd alura-agent
cp .env.example .env
nano .env   # preencha GOOGLE_API_KEY
```
 
### 5. Construir e rodar o container
 
```bash
docker build -t alura-agent .
docker run -d --name alura-agent \
  --env-file .env \
  -p 8501:8501 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  alura-agent
```
 
### 6. Indexar o documento dentro do container (primeira vez)
 
```bash
docker exec -it alura-agent python ingest.py --file regras_gamificacao.csv
```
 
### 7. Acessar publicamente
 
```
http://SEU_IP_PUBLICO:8501
```
 
> 📸 **Print/link da aplicação rodando na OCI:** _[inserir aqui a captura de
> tela ou o link público após o deploy]_
 
---
 
## 📁 Estrutura do repositório
 
```
alura-agent/
├── app.py                  # Interface Streamlit (chat, fontes, histórico)
├── ingest.py                # Ingestão: extração, limpeza, chunking, indexação
├── rag_chain.py              # Pipeline RAG: busca, prompt, chamada ao Gemini, fallback
├── requirements.txt          # Dependências Python
├── Dockerfile                 # Empacotamento para deploy
├── .env.example                # Modelo de variáveis de ambiente
├── .gitignore
├── regras_gamificacao.csv       # Documento de exemplo indexado
└── README.md
```
 
---
 
## 🔭 Próximos passos / melhorias futuras
 
- **Monitoramento:** registrar logs estruturados (JSON Lines) de cada
  pergunta/resposta/fontes/tempo de resposta, com um dashboard simples lido
  diretamente dos logs.
- **CI/CD:** pipeline no GitHub Actions para rodar testes e publicar a imagem
  Docker automaticamente a cada push na branch principal.
- **Atualização automática de documentos:** rotina periódica (cron ou GitHub
  Action agendada) que reprocessa documentos alterados e atualiza o índice
  Chroma sem downtime.
- **Feedback dos usuários:** persistir os cliques de 👍/👎 já presentes na
  interface para identificar perguntas mal respondidas.
