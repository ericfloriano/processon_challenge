# Agente EDA para Análise de CSV

Uma aplicação web interativa para Análise Exploratória de Dados (EDA) automatizada, construída com Gradio e potencializada pelo Google Gemini. Esta ferramenta permite que usuários façam upload de qualquer arquivo CSV e obtenham insights através de relatórios textuais e perguntas em linguagem natural que podem gerar gráficos dinamicamente.

**[Acesse a Aplicação Online Aqui](https://huggingface.co/spaces/ericbueno/agente-eda-csv)**

## Funcionalidades Principais

- **Upload Genérico de CSV:** Carrega qualquer arquivo CSV, tratando automaticamente diferentes delimitadores (`,`, `;`, `|`, etc.), múltiplas codificações de caracteres (`utf-8`, `latin1`) e limpando nomes de colunas complexos.
- **Análise Completa (Apenas Texto):** Gera um relatório textual inicial com estatísticas descritivas e análise de valores nulos sobre uma amostra dos dados, ideal para um primeiro diagnóstico rápido.
- **Perguntas Específicas com Gráficos:** Permite que o usuário faça perguntas em linguagem natural. O agente é capaz de interpretar pedidos de visualização e gerar dinamicamente histogramas, boxplots e gráficos de dispersão.
- **Interface Intuitiva:** Apresenta respostas e gráficos lado a lado e inclui uma função para "Limpar Sessão", permitindo que o usuário reinicie a análise com um novo arquivo de forma fluida e sem precisar recarregar a página.

## Arquitetura e Tecnologias

Este projeto utiliza uma arquitetura híbrida **"Python como Analista, Agente como Curador"**:
- **Backend (Python):** Executa toda a manipulação de dados com `Pandas` e a geração de gráficos com `Matplotlib`/`Seaborn`.
- **Agente (Google Gemini):** Atua como o "cérebro" da operação, interpretando as perguntas do usuário em linguagem natural, planejando as ações (como qual gráfico gerar) e gerando relatórios e respostas textuais.
- **Interface:** `Gradio`.
- **Orquestração do Agente:** `Agno Framework`.

## Como Usar a Aplicação

1.  Acesse a aba **"📁 Carregar Dataset"**.
2.  Selecione um arquivo `.csv` do seu computador. **Importante:** Aguarde o upload completar 100% antes de clicar em "📊 Carregar e Analisar".
3.  Para um relatório textual inicial, vá para a aba **"🔍 Análise Completa"**.
4.  Para investigações detalhadas e geração de gráficos, vá para **"❓ Perguntas Específicas"** e digite seu pedido (ex: "crie um boxplot para a coluna 'Idade'").
5.  Os gráficos gerados aparecerão na galeria ao lado. **Clique nos gráficos para expandi-los**.
6.  Para começar de novo, clique em **"🧹 Limpar Sessão e Iniciar Nova Análise"** na primeira aba.

## Como Executar Localmente

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/ericfloriano/processon_challenge.git]
    cd [NOME_DA_PASTA]
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure sua chave de API:**
    - Crie um arquivo chamado `.env` na raiz do projeto.
    - Adicione a seguinte linha, substituindo pela sua chave:
        ```
        GOOGLE_API_KEY=sua_chave_google_aqui
        ```

5.  **Execute a aplicação:**
    ```bash
    python app.py
    ```
    A aplicação estará disponível em `http://localhost:7860`.