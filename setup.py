"""
Script de configuração do ambiente para o Agente EDA - VERSÃO GRADIO + GEMINI
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Instala as dependências necessárias"""
    print("📦 Instalando dependências...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def setup_environment():
    """Configura o ambiente"""
    print("🔧 Configurando ambiente...")
    
    # Criar diretórios necessários
    dirs = ["app", "relatorios_eda", "graficos_eda", "dados_csv"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Diretório criado: {dir_name}")
    
    # Criar arquivo .env se não existir
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Configurações do Agente EDA - GRADIO + GEMINI
GOOGLE_API_KEY=sua_chave_google_aqui

# Configurações do Gradio
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# Configurações opcionais
AGNO_LOG_LEVEL=INFO
AGNO_DEBUG=False

# Configurações do banco de dados
DB_FILE=eda_agent.db
"""
        env_file.write_text(env_content)
        print("📝 Arquivo .env criado")
    
    print("✅ Ambiente configurado!")

def check_api_key():
    """Verifica se a chave da API está configurada"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "sua_chave_google_aqui":
        print("⚠️  ATENÇÃO: Configure sua GOOGLE_API_KEY!")
        print("1. Obtenha sua chave em: https://ai.google.dev/gemini-api/docs/api-key")
        print("2. Configure com: export GOOGLE_API_KEY=sua_chave_aqui")
        print("3. Ou edite o arquivo .env")
        return False
    
    print("✅ Chave da API configurada!")
    return True

def test_gradio():
    """Testa se o Gradio está funcionando"""
    print("🧪 Testando Gradio...")
    
    try:
        import gradio as gr
        print("✅ Gradio instalado e funcionando!")
        return True
    except ImportError:
        print("❌ Erro: Gradio não está instalado")
        return False

def test_gemini():
    """Testa se o Gemini está funcionando"""
    print("🧪 Testando conexão com Gemini...")
    
    try:
        from agno.agent import Agent
        from agno.models.google import Gemini
        
        agent = Agent(model=Gemini(id="gemini-2.0-flash"))
        response = agent.run("Responda apenas: 'Gemini funcionando!'")
        
        if "funcionando" in response.content.lower():
            print("✅ Gemini funcionando corretamente!")
            return True
        else:
            print("⚠️  Gemini respondeu, mas resposta inesperada")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar Gemini: {e}")
        return False

def main():
    """Função principal de setup"""
    print("🚀 CONFIGURAÇÃO DO AGENTE EDA - GRADIO + GEMINI")
    print("=" * 60)
    
    # Instalar dependências
    if not install_requirements():
        return
    
    # Configurar ambiente
    setup_environment()
    
    # Testar Gradio
    test_gradio()
    
    # Verificar chave da API
    if check_api_key():
        # Testar Gemini
        test_gemini()
    
    print("\n🎉 CONFIGURAÇÃO CONCLUÍDA!")
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. Se ainda não o fez, configure sua GOOGLE_API_KEY no arquivo .env")
    print("2. Execute: python app.py")
    print("3. Acesse: http://localhost:7860")
    print("4. Ou use: gradio app.py")

if __name__ == "__main__":
    main()