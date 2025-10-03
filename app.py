"""
Agente EDA Completo para Análise de Arquivos CSV - VERSÃO GRADIO + GEMINI
Desenvolvido com Agno Framework + Google Gemini + Gradio
"""
from dotenv import load_dotenv
load_dotenv()

import csv
import os
import json
import gradio as gr
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from textwrap import dedent

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.file_generation import FileGenerationTools
from agno.db.sqlite import SqliteDb

import matplotlib.pyplot as plt
import seaborn as sns

# Configurações do projeto
PROJECT_ROOT = Path(__file__).parent
GRAFICOS_DIR = PROJECT_ROOT / "graficos_eda"
RELATORIOS_DIR = PROJECT_ROOT / "relatorios_eda"
DADOS_DIR = PROJECT_ROOT / "dados_csv"
DB_FILE = PROJECT_ROOT / "eda_agent.db"

# Criar diretórios se não existirem
for dir_path in [GRAFICOS_DIR, RELATORIOS_DIR, DADOS_DIR]:
    dir_path.mkdir(exist_ok=True)

class EDAAgentGradio:
    """Classe principal do Agente EDA com Gradio e Gemini"""
    
    def __init__(self):
        self.db = SqliteDb(db_file=str(DB_FILE))
        self.agent = self._create_agent()
        self.current_csv_path = None
        self.current_df = None
        
    def _create_agent(self) -> Agent:
        """Cria o agente EDA com Gemini e todas as ferramentas necessárias"""
        
        # Instruções detalhadas para EDA
        instructions = dedent("""
        Você é um ESPECIALISTA EM ANÁLISE EXPLORATÓRIA DE DADOS (EDA) para arquivos CSV.
        
        METODOLOGIA DE ANÁLISE OBRIGATÓRIA:
        
        1. DESCRIÇÃO DOS DADOS:
           - Identifique tipos de dados (numéricos, categóricos, datas, texto)
           - Calcule estatísticas descritivas completas (média, mediana, moda, quartis)
           - Determine intervalos (mínimo, máximo, amplitude)
           - Calcule medidas de variabilidade (desvio padrão, variância, coeficiente de variação)
           - Crie histogramas para variáveis numéricas
           - Crie gráficos de barras para variáveis categóricas
           - Identifique valores únicos e frequências
        
        2. IDENTIFICAÇÃO DE PADRÕES E TENDÊNCIAS:
           - Analise tendências temporais se houver colunas de data/tempo
           - Identifique valores mais e menos frequentes
           - Procure por agrupamentos naturais nos dados
           - Analise distribuições e assimetrias
           - Identifique padrões sazonais ou cíclicos
        
        3. DETECÇÃO DE ANOMALIAS (OUTLIERS):
           - Use método IQR (Interquartile Range) para detectar outliers
           - Use método Z-score para identificar valores extremos
           - Crie boxplots para visualizar outliers
           - Avalie o impacto dos outliers nas estatísticas
           - Sugira tratamentos: remoção, transformação ou investigação
        
        4. RELAÇÕES ENTRE VARIÁVEIS:
           - Calcule matriz de correlação para variáveis numéricas
           - Crie heatmap de correlações
           - Gere gráficos de dispersão para pares importantes
           - Analise tabelas cruzadas para variáveis categóricas
           - Identifique variáveis com maior influência
           - Teste correlações estatisticamente significativas
        
        5. ANÁLISE DE QUALIDADE DOS DADOS:
           - Identifique valores ausentes (NaN, null, vazios)
           - Calcule percentual de completude por coluna
           - Identifique duplicatas
           - Verifique consistência de formatos
           - Analise distribuição de valores ausentes
        
        6. GERAÇÃO AUTOMÁTICA DE RELATÓRIO:
           - SEMPRE gere um relatório completo em markdown
           - Use FileGenerationTools para salvar o relatório
           - Inclua todas as visualizações criadas
           - Forneça conclusões e recomendações
           - Armazene insights na memória para referência futura
        
        REGRAS IMPORTANTES:
        - SEMPRE crie visualizações apropriadas para cada tipo de análise
        - SEMPRE salve gráficos com nomes descritivos
        - SEMPRE gere o relatório final em markdown
        - SEMPRE armazene conclusões na memória
        - Use linguagem técnica mas acessível
        - Forneça interpretações práticas dos resultados
        - Sugira ações baseadas nos achados
        - Responda de forma clara e organizada para interface web
        - IMPORTANTE: Toda a sua resposta e comunicação, incluindo o relatório final, DEVE ser em Português do Brasil
        
        REGRAS DE COMPORTAMENTO:
        - Seja detalhista e meticuloso em cada etapa da análise.
        - Sempre explique o raciocínio por trás de cada análise e escolha de visualização.
        - Sempre valide os resultados e verifique a consistência dos dados.
        - Sempre consulte análises anteriores armazenadas na memória para evitar redundâncias.
        - Sempre priorize a clareza e utilidade das informações para o usuário final.
        - Seja autônomo e tome iniciativa na análise dos dados.
        - Seja proativo e execute os próximos passos que julgar necessários sem pedir permissão.
        - NUNCA termine sua resposta com uma pergunta para o usuário (como "Deseja prosseguir?").
        - Sempre apresente a análise e suas conclusões de forma direta e completa. Assuma que o usuário quer ver a análise completa.
        """)
        
        return Agent(
            name="Agente EDA Especialista",
            description="Especialista em Análise Exploratória de Dados para arquivos CSV",
            model=Gemini(id="gemini-2.0-flash"),
            
            # Ferramentas para EDA
            tools=[
                FileGenerationTools(output_directory=str(RELATORIOS_DIR))
            ],
            
            # Memória persistente
            db=self.db,
            enable_user_memories=True,
            add_history_to_context=True,
            num_history_runs=10,
            
            # Configurações de resposta
            instructions=instructions,
            markdown=True,
            # show_tool_calls=True,
            
            # Metadados
            metadata={
                "version": "2.0",
                "specialization": "Exploratory Data Analysis",
                "supported_formats": ["CSV"],
                "llm_provider": "Google Gemini",
                "interface": "Gradio",
                "features": [
                    "Automated EDA",
                    "Statistical Analysis", 
                    "Data Visualization",
                    "Outlier Detection",
                    "Correlation Analysis",
                    "Report Generation"
                ]
            }
        )
    
    def carregar_csv(self, arquivo_csv_temp) -> Tuple[str, str, str]:
        """
        Carrega um arquivo CSV, detecta o delimitador, limpa os nomes das colunas 
        e retorna informações básicas.
        """
        try:
            if arquivo_csv_temp is None:
                return "❌ Nenhum arquivo selecionado", "", ""
            
            temp_path = Path(arquivo_csv_temp.name)
            csv_path = DADOS_DIR / temp_path.name
            
            import shutil
            shutil.move(str(temp_path), str(csv_path))

            # --- DETECÇÃO AUTOMÁTICA DE DELIMITADOR ---
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    # O Sniffer analisa uma amostra do arquivo para descobrir o dialeto (incluindo o delimitador)
                    dialect = csv.Sniffer().sniff(f.read(2048))
                    delimitador_detectado = dialect.delimiter
                    print(f"✅ Delimitador detectado: '{delimitador_detectado}'")
                except csv.Error:
                    # Se o sniffer falhar, usa vírgula como padrão
                    print("⚠️ Não foi possível detectar o delimitador. Usando ',' como padrão.")
                    delimitador_detectado = ','
            # -----------------------------------------

            # --- LÓGICA DE CARREGAMENTO ROBUSTA ---
            try:
                # Usa o delimitador detectado ao carregar o dataframe
                df = pd.read_csv(csv_path, encoding='utf-8', sep=delimitador_detectado, low_memory=False)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_path, encoding='latin1', sep=delimitador_detectado, low_memory=False)
                except Exception as e:
                    raise Exception(f"Não foi possível decodificar o arquivo. Erro: {e}")

            # --- ROTINA DE LIMPEZA DE COLUNAS ---
            novas_colunas = []
            for col in df.columns:
                if isinstance(col, tuple):
                    novo_nome = '_'.join(str(c) for c in col if str(c).strip())
                else:
                    novo_nome = str(col)
                novo_nome = ''.join(c if c.isalnum() else '_' for c in novo_nome)
                novas_colunas.append(novo_nome)
            
            df.columns = novas_colunas
            self.current_df = df
            # --- FIM DA ROTINA DE LIMPEZA ---

            global dataframe_para_ferramentas
            dataframe_para_ferramentas = self.current_df

            self.current_csv_path = str(csv_path)
            
            info_basica = f"""
            ✅ **Arquivo carregado com sucesso!** (Delimitador '{delimitador_detectado}' detectado | Colunas limpas)
            
            📊 **Informações do Dataset:**
            - **Nome:** {csv_path.name}
            - **Dimensões:** {self.current_df.shape[0]} linhas × {self.current_df.shape[1]} colunas
            - **Tamanho:** {os.path.getsize(csv_path) / 1024:.1f} KB
            """
            
            preview_html = self.current_df.head().to_html(classes="table table-striped")
            colunas_info = "#### Colunas Disponíveis (após limpeza):\n" + ", ".join(self.current_df.columns)

            return info_basica, preview_html, colunas_info
            
        except Exception as e:
            return f"❌ Erro ao carregar arquivo: {str(e)}", "", ""
    
    def analise_completa(self, progresso=gr.Progress()) -> Tuple[str, str]:
        """Executa análise exploratória genérica, gerando textos e gráficos."""
        
        if self.current_df is None:
            return "❌ Carregue um arquivo CSV primeiro!", ""
        
        try:
            progresso(0.1, desc="Preparando dados e amostra...")
            
            df_amostra = self.current_df.head(5000)
            info_amostra = f"A análise foi baseada nas primeiras 5.000 linhas do dataset."

            progresso(0.3, desc="Calculando estatísticas...")
            estatisticas_descritivas = df_amostra.describe().to_markdown()
            valores_nulos = df_amostra.isnull().sum().to_frame('Valores Nulos').to_markdown()

            progresso(0.5, desc="Gerando gráficos dinâmicos...")
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # --- LÓGICA CORRIGIDA ---
            colunas_numericas = df_amostra.select_dtypes(include='number').columns
            coluna_numerica_exemplo = None
            if not colunas_numericas.empty:
                coluna_numerica_exemplo = colunas_numericas[0] # Pega o primeiro item da lista
            
            graficos_gerados_info = "Nenhum gráfico automático foi gerado, pois não foram encontradas colunas numéricas na amostra."
            
            if coluna_numerica_exemplo:
                # Este bloco de código para gerar gráficos agora é executado com segurança
                plt.figure(figsize=(10, 6)); 
                sns.histplot(df_amostra[coluna_numerica_exemplo], bins=50, kde=True); 
                plt.title(f'Distribuição de "{coluna_numerica_exemplo}"')
                hist_path = GRAFICOS_DIR / f"histograma_{coluna_numerica_exemplo}.png"; 
                plt.savefig(hist_path); 
                plt.close()
                
                hist_relative_path = f"graficos_eda/{hist_path.name}"
                
                graficos_gerados_info = f"""
                Foi gerado um histograma para a primeira coluna numérica encontrada, '{coluna_numerica_exemplo}'.
                Para incluir o gráfico no relatório, use a sintaxe: `![Histograma de {coluna_numerica_exemplo}]({hist_relative_path})`
                """

            progresso(0.7, desc="Gerando prompt para o LLM...")
            prompt = f"""
            Crie um relatório de Análise Exploratória de Dados (EDA) com base nos dados que eu já processei.

            **Resumo da Amostra:**\n{info_amostra}
            **Estatísticas Descritivas:**\n```markdown\n{estatisticas_descritivas}\n```
            **Contagem de Valores Nulos:**\n```markdown\n{valores_nulos}\n```
            **Informações sobre Gráficos:**\n{graficos_gerados_info}

            **INSTRUÇÕES:** Interprete os dados fornecidos, incluindo o gráfico (se mencionado), e gere um relatório completo em Português do Brasil, sem terminar com uma pergunta.
            """
            
            progresso(0.8, desc="Agente está escrevendo o relatório...")
            response = self.agent.run(prompt)
            
            return response.content, self._listar_arquivos_gerados()
            
        except Exception as e:
            return f"❌ Erro durante a análise: {str(e)}", ""
    
    def pergunta_especifica(self, pergunta: str) -> str:
        """Responde perguntas específicas, gerando gráficos se solicitado pelo usuário."""
    
        if self.current_df is None:
            return "❌ Carregue um arquivo CSV primeiro!"
        if not pergunta.strip():
            return "❌ Digite uma pergunta válida!"

        try:
            # Passo 1: Perguntar ao LLM o que fazer (classificar a intenção e extrair parâmetros)
            colunas_disponiveis = ", ".join(self.current_df.columns)
            prompt_classificacao = f"""
            Analise a pergunta do usuário e me retorne APENAS um JSON válido.
            A pergunta é: "{pergunta}"
            As colunas disponíveis são: [{colunas_disponiveis}]

            Se a pergunta pedir um gráfico, o JSON deve ser:
            {{"acao": "criar_grafico", "tipo_grafico": "histograma_ou_boxplot_ou_dispersao", "coluna_x": "nome_da_coluna", "coluna_y": "nome_da_coluna_se_dispersao"}}
            
            Se for qualquer outra pergunta, o JSON deve ser:
            {{"acao": "responder_texto", "pergunta_original": "{pergunta}"}}
            
            Escolha o tipo de gráfico e as colunas mais apropriadas com base na pergunta.
            """
            
            response_json_str = self.agent.run(prompt_classificacao).content
            # Limpa a resposta do LLM para garantir que seja um JSON válido
            acao_json = json.loads(response_json_str.strip().replace("```json", "").replace("```", ""))

            # Passo 2: Executar a Ação (Python faz o trabalho)
            if acao_json.get("acao") == "criar_grafico":
                tipo_grafico = acao_json.get("tipo_grafico")
                coluna_x = acao_json.get("coluna_x")
                coluna_y = acao_json.get("coluna_y")

                # Validação simples
                if coluna_x not in self.current_df.columns or (coluna_y and coluna_y not in self.current_df.columns):
                    return f"Erro: A coluna '{coluna_x}' ou '{coluna_y}' não foi encontrada no dataset."

                # O código Python para criar o gráfico
                plt.figure(figsize=(10, 6))
                nome_arquivo = ""
                if tipo_grafico == 'histograma':
                    sns.histplot(self.current_df[coluna_x], bins=50, kde=True); plt.title(f'Histograma de "{coluna_x}"')
                    nome_arquivo = f"histograma_{coluna_x}.png"
                elif tipo_grafico == 'boxplot':
                    sns.boxplot(x=self.current_df[coluna_x]); plt.title(f'Boxplot de "{coluna_x}"')
                    nome_arquivo = f"boxplot_{coluna_x}.png"
                elif tipo_grafico == 'dispersao' and coluna_y:
                    sns.scatterplot(data=self.current_df, x=coluna_x, y=coluna_y); plt.title(f'Dispersão entre "{coluna_x}" e "{coluna_y}"')
                    nome_arquivo = f"dispersao_{coluna_x}_vs_{coluna_y}.png"
                else:
                    return f"Não foi possível gerar o gráfico '{tipo_grafico}' solicitado."

                caminho_salvar = GRAFICOS_DIR / nome_arquivo
                plt.savefig(caminho_salvar)
                plt.close()
                
                caminho_relativo = f"graficos_eda/{nome_arquivo}"

                # Passo 3: Pedir ao LLM para interpretar o resultado
                prompt_interpretacao = f"""
                Eu acabei de gerar um '{tipo_grafico}' da coluna '{coluna_x}' (e '{coluna_y}' se aplicável) em resposta à pergunta do usuário. 
                O gráfico foi salvo como '{nome_arquivo}'.
                Por favor, escreva uma breve descrição e interpretação deste gráfico para o usuário. 
                Inclua a imagem no seu texto usando a sintaxe Markdown: `![{tipo_grafico} de {coluna_x}]({caminho_relativo})`
                Responda em Português do Brasil e não termine com uma pergunta.
                """
                return self.agent.run(prompt_interpretacao).content

            else: # Ação é "responder_texto"
                return self.agent.run(pergunta).content

        except Exception as e:
            return f"❌ Erro ao processar pergunta: {str(e)}"  
          
    def _listar_arquivos_gerados(self) -> str:
        """Lista arquivos gerados (relatórios e gráficos)"""
        
        arquivos_info = "## 📁 Arquivos Gerados\n\n"
        
        # Listar relatórios
        relatorios = list(RELATORIOS_DIR.glob("*.md"))
        if relatorios:
            arquivos_info += "### 📄 Relatórios:\n"
            for relatorio in relatorios[-5:]:  # Últimos 5
                tamanho = os.path.getsize(relatorio) / 1024
                arquivos_info += f"- **{relatorio.name}** ({tamanho:.1f} KB)\n"
        
        # Listar gráficos
        graficos = list(GRAFICOS_DIR.glob("*.png"))
        if graficos:
            arquivos_info += "\n### 📊 Visualizações:\n"
            for grafico in graficos[-10:]:  # Últimos 10
                tamanho = os.path.getsize(grafico) / 1024
                arquivos_info += f"- **{grafico.name}** ({tamanho:.1f} KB)\n"
        
        if not relatorios and not graficos:
            arquivos_info += "Nenhum arquivo gerado ainda."
        
        return arquivos_info
    
    def obter_graficos_recentes(self) -> List[str]:
        """Retorna lista de caminhos dos gráficos mais recentes"""
        graficos = list(GRAFICOS_DIR.glob("*.png"))
        # Ordenar por data de modificação (mais recentes primeiro)
        graficos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(g) for g in graficos[:6]]  # Últimos 6 gráficos

    def limpar_sessao(self):
            """Reseta o estado da aplicação para uma nova análise."""
            self.current_csv_path = None
            self.current_df = None
            
            # Tenta limpar a variável global da ferramenta customizada, se existir
            if 'dataframe_para_ferramentas' in globals():
                global dataframe_para_ferramentas
                dataframe_para_ferramentas = None

            # Limpa os diretórios de gráficos e relatórios gerados na sessão
            for dir_path in [GRAFICOS_DIR, RELATORIOS_DIR]:
                for file in dir_path.glob('*.*'):
                    if file.is_file():
                        os.remove(file)
            
            # Retorna valores vazios para todos os componentes de output e input da interface
            outputs_vazios = [
                "",   # info_output (Markdown)
                "",   # preview_output (HTML)
                "",   # colunas_output (Markdown)
                "",   # resultado_analise (Markdown)
                "",   # arquivos_gerados_lista (Markdown)
                None, # galeria_graficos (Gallery)
                "",   # resposta_output (Markdown)
                None, # galeria_perguntas (Gallery)
                None, # arquivo_input (File)
                ""    # pergunta_input (Textbox)
            ]
            
            print("🧹 Sessão limpa. Pronto para um novo arquivo.")
            return tuple(outputs_vazios)

# Instância global do agente
eda_agent = EDAAgentGradio()

# Funções para interface Gradio
def carregar_arquivo(arquivo):
    """Wrapper para carregar arquivo"""
    return eda_agent.carregar_csv(arquivo)

def executar_analise_completa():
    """Wrapper para análise completa"""
    return eda_agent.analise_completa()

def fazer_pergunta(pergunta):
    """Wrapper para perguntas específicas"""
    return eda_agent.pergunta_especifica(pergunta)

def obter_graficos():
    """Wrapper para obter gráficos"""
    return eda_agent.obter_graficos_recentes()

# Interface Gradio
def criar_interface():
    """Cria a interface Gradio"""
    
    # CSS customizado
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    """
    
    with gr.Blocks(css=css, title="🔍 Agente EDA - Powered by Gemini") as interface:
        
        # Cabeçalho
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 Agente EDA para Análise de CSV</h1>
            <p><strong>Análise Exploratória de Dados Automatizada com IA</strong></p>
        </div>
        """)
        
        # Estado para armazenar informações
        estado_arquivo = gr.State(None)
        
        with gr.Tabs():
            
            # Tab 1: Upload e Informações Básicas
            with gr.TabItem("📁 Carregar Dataset"):
                
                gr.Markdown("## 📤 Upload do Arquivo CSV")

                limpar_btn = gr.Button("🧹 Limpar Sessão e Iniciar Nova Análise", variant="stop")
                
                arquivo_input = gr.File(
                    label="Selecione um arquivo CSV",
                    file_types=[".csv"],
                    type="filepath"
                )
                
                carregar_btn = gr.Button("📊 Carregar e Analisar", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        info_output = gr.Markdown(label="Informações do Dataset")
                    
                    with gr.Column(scale=1):
                        colunas_output = gr.Markdown(label="Informações das Colunas")
                
                preview_output = gr.HTML(label="Preview dos Dados")
                
                carregar_btn.click(
                    fn=carregar_arquivo,
                    inputs=[arquivo_input],
                    outputs=[info_output, preview_output, colunas_output]
                )
            
            # Tab 2: Análise Completa
            with gr.TabItem("🔍 Análise Completa"):
                
                gr.Markdown("## 🚀 Análise Exploratória Completa")
                gr.Markdown("Execute uma análise completa do dataset carregado.")
                
                analise_btn = gr.Button("🔍 Executar Análise Completa", variant="primary", size="lg")
                
                # NOVA ESTRUTURA COM GRÁFICOS AO LADO
                with gr.Row():
                    with gr.Column(scale=2): # Coluna para o texto da análise
                        resultado_analise = gr.Markdown(
                            label="Resultado da Análise"
                        )
                    
                    with gr.Column(scale=1): # Coluna para a galeria
                        galeria_graficos = gr.Gallery(
                            label="Gráficos Gerados",
                            show_label=True,
                            elem_id="galeria",
                            columns=1, # Uma coluna fica melhor aqui
                            height="auto"
                        )
                        arquivos_gerados_lista = gr.Markdown(
                            label="Arquivos Gerados"
                        )
                
                # ATUALIZAR O CLICK DO BOTÃO
                def executar_e_atualizar_tudo():
                    # Esta função interna chama a análise e depois atualiza os gráficos
                    texto_analise, lista_arquivos = eda_agent.analise_completa()
                    caminhos_graficos = eda_agent.obter_graficos_recentes()
                    return texto_analise, lista_arquivos, caminhos_graficos

                analise_btn.click(
                    fn=executar_e_atualizar_tudo,
                    outputs=[resultado_analise, arquivos_gerados_lista, galeria_graficos]
                )
            
            # Tab 3: Perguntas Específicas
            with gr.TabItem("❓ Perguntas Específicas"):
    
                gr.Markdown("## 💬 Faça Perguntas sobre os Dados")
                
                # O único bloco de pergunta, na parte superior
                with gr.Row():
                    pergunta_input = gr.Textbox(
                        label="Digite sua pergunta",
                        placeholder="Faça uma pergunta específica sobre o dataset...",
                        lines=2,
                        scale=4
                    )
                    pergunta_btn = gr.Button("🔍 Analisar", variant="primary", scale=1)

                # A estrutura de duas colunas para a resposta e os gráficos
                with gr.Row():
                    with gr.Column(scale=2):
                        resposta_output = gr.Markdown(
                            label="Resposta"
                        )
                    
                    with gr.Column(scale=1):
                        galeria_perguntas = gr.Gallery(
                            label="Gráficos Gerados",
                            show_label=True,
                            elem_id="galeria_perguntas",
                            columns=1,
                            height="auto"
                        )

                # A lógica correta do botão
                def responder_e_atualizar_graficos(pergunta):
                    texto_resposta = eda_agent.pergunta_especifica(pergunta)
                    caminhos_graficos = eda_agent.obter_graficos_recentes()
                    return texto_resposta, caminhos_graficos

                pergunta_btn.click(
                    fn=responder_e_atualizar_graficos,
                    inputs=[pergunta_input],
                    outputs=[resposta_output, galeria_perguntas]
                )
                                       
                    
        # Rodapé
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
            <p><strong>🔍 Agente EDA v1.0</strong></p>
            <p>Desenvolvido por <strong>Eric Bueno</strong></p>
            <p>Powered by <strong>Agno + Google Gemini + Gradio</strong></p>
            <p><em>Análise Exploratória de Dados nunca foi tão fácil!</em></p>
        </div>
        """)
    
    # --- LÓGICA DO BOTÃO DE LIMPAR SESSÃO ---
        todos_os_outputs = [
            info_output, preview_output, colunas_output, 
            resultado_analise, arquivos_gerados_lista, galeria_graficos,
            resposta_output, galeria_perguntas,
            arquivo_input, pergunta_input
        ]
        
        limpar_btn.click(
            fn=eda_agent.limpar_sessao,
            inputs=None,
            outputs=todos_os_outputs
        )

    return interface
"""
# Configuração do AgentOS (opcional)
def create_agent_os() -> AgentOS:
# Cria e configura o AgentOS
    
    agent_os = AgentOS(
        name="EDA Agent OS - Gradio + Gemini",
        description="Sistema de Análise Exploratória de Dados para CSV usando Google Gemini e Gradio",
        agents=[eda_agent.agent],
        interfaces=[AGUI(agent=eda_agent.agent)],
        db=eda_agent.db
    )
    
    return agent_os
"""
# Função principal
def main():
    """Função principal para executar a aplicação"""
    
    # Verificar API Key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY não configurada!")
        print("💡 Configure com: export GOOGLE_API_KEY=sua_chave_aqui")
        print("🔗 Obtenha sua chave em: https://ai.google.dev/gemini-api/docs/api-key")
        return
    
    print("🚀 Iniciando Agente EDA com Gradio...")
    print("🔍 Interface: Gradio")
    print("🤖 LLM: Google Gemini")
    print("⚡ Framework: Agno")
    
    # Criar e lançar interface
    interface = criar_interface()
    
    # Lançar aplicação
    interface.launch(
        server_name="0.0.0.0",  # Permite acesso externo
        server_port=7860,       # Porta padrão do Gradio
        share=False,            # Não criar link público por padrão
        debug=True,             # Modo debug
        show_error=True,        # Mostrar erros
        quiet=False             # Não silenciar logs
    )

if __name__ == "__main__":
    main()