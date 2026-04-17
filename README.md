# 🎬 Video Viral Agent

Sistema multi-agent que identifica e extrai automaticamente momentos virais de vídeos usando IA.

## 🎯 O que ele faz

O Video Viral Agent analisa vídeos longos e extrai automaticamente os momentos que correspondem aos seus critérios específicos, como:

- Momentos engraçados e cômicos
- Partes onde fala sobre determinado assunto
- Citações de pessoas específicas
- Momentos de alto impacto emocional
- E qualquer outro critério que você definir

**Como funciona:**
1. Transcreve o áudio do vídeo (local ou via API)
2. Identifica os momentos que batem com seu critério
3. Refina os cortes para não cortar palavras no meio
4. Gera clips de vídeo com os momentos identificados

## 🚀 Tecnologias Utilizadas

### Core
- **Python 3.12+** - Linguagem principal
- **LangGraph** - Orquestração do workflow multi-agent
- **LangChain** - Framework para LLMs
- **Pydantic** - Validação de dados e type-safety

### Transcrição de Áudio
- **faster-whisper** - Transcrição local com CTranslate2 (4x mais rápido)
- **Groq API** - Transcrição via API com Whisper Large v3 Turbo
- **Whisper** - Modelo de reconhecimento de voz da OpenAI

### Edição de Vídeo
- **MoviePy 2.0** - Manipulação e edição de vídeos
- **FFmpeg** - Processamento de vídeo (backend)

### IA / LLM
- **OpenRouter** - Acesso a múltiplos modelos LLM
- **Groq** - API para transcrição de áudio
- Modelos suportados: GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash, etc.

### Web Interface
- **Flask** - Servidor web
- **Bootstrap 5** - Interface responsiva
- **JavaScript** - Upload assíncrono e polling de status

## 📦 Instalação

### Pré-requisitos
- Python 3.12 ou superior
- FFmpeg instalado no sistema
- UV (recomendado) ou pip

### Passo 1: Clone o repositório
```bash
git clone <seu-repositorio>
cd video_viral_agent
```

### Passo 2: Instale as dependências
```bash
# Com UV (recomendado)
pip install uv
uv sync

# Ou com pip
pip install -e .
```

### Passo 3: Configure as variáveis de ambiente
```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas chaves de API:

```bash
# Modo de transcrição: local (grátis) ou groq (API)
AUDIO_TRANSCRIPTION_MODE=local

# API Keys (opcional, apenas se usar Groq)
GROQ_API_KEY=sua_chave_groq_aqui
GROQ_WHISPER_MODEL=whisper-large-v3-turbo

# OpenRouter (obrigatório para análise)
OPENROUTER_API_KEY=sua_chave_openrouter_aqui
VLM_MODEL_NAME=google/gemini-2.0-flash-lite-001
LLM_MODEL_NAME=google/gemini-2.0-flash-lite-001
```

## 🎮 Como Usar

### Opção 1: Interface Web (Recomendado)

#### Inicie o servidor Flask:
```bash
uv run python app.py
```

#### Acesse no navegador:
```
http://localhost:5000
```

#### Passos:
1. Clique na área de upload ou arraste seu vídeo
2. Descreva o critério de análise (ex: "Capture partes onde ele cita desenvolvedores plenos")
3. Clique em "Processar Vídeo"
4. Aguarde o processamento (1-5 minutos dependendo do tamanho)
5. Assista e faça download dos clips gerados

**Formatos suportados:** MP4, MOV, AVI, MKV (máx. 500MB)

### Opção 2: API Python

```python
import asyncio
from src.workflow import run_workflow
from src.state import VideoAnalysisState, AnalysisStatus

async def process_video():
    # Defina o estado inicial
    initial_state = VideoAnalysisState(
        videoPath="caminho/do/video.mp4",
        analysis=[
            "Capture partes onde ele cita desenvolvedores plenos",
            "Momentos engraçados e cômicos"
        ],
        status=AnalysisStatus.PENDING
    )
    
    # Execute o workflow
    result = await run_workflow(initial_state)
    
    # Verifique os resultados
    print(f"Clips gerados: {result.get('outputClips', [])}")
    print(f"Transcrição: {result.get('transcription', '')}")

# Execute
asyncio.run(process_video())
```

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Transcrição de Áudio                                     │
│     ├─ Local: faster-whisper (CTranslate2)                   │
│     └─ Cloud: Groq API (Whisper Large v3)                   │
│         │                                                    │
│         v                                                    │
│  2. Identificação de Momentos                                │
│     └─ LLM analisa transcrição e encontra momentos           │
│         │                                                    │
│         v                                                    │
│  3. Refinamento de Contexto                                  │
│     └─ Expande cortes para evitar cortes no meio da frase   │
│         │                                                    │
│         v                                                    │
│  4. Edição de Vídeo                                          │
│     └─ MoviePy corta e exporta os clips                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Estrutura de Arquivos

```
video_viral_agent/
├── app.py                    # Servidor Flask
├── templates/
│   └── index.html           # Interface web
├── src/
│   ├── nodes/
│   │   ├── transcribe_audio.py      # Transcrição (local/Groq)
│   │   ├── identify_moments.py      # Identificação com LLM
│   │   ├── refine_clip_context.py   # Refinamento de cortes
│   │   └── edit_video.py            # Edição de vídeo
│   ├── agents.py             # Configuração de LLMs
│   ├── state.py              # Estado da aplicação
│   └── workflow.py           # Orquestração LangGraph
├── uploads/                  # Vídeos enviados (temporário)
├── output_clips/             # Clips processados
└── .env                      # Variáveis de ambiente
```

## 🎼 Modos de Transcrição

### Local Mode (Recomendado - Grátis)
```bash
AUDIO_TRANSCRIPTION_MODE=local
```
- **Vantagens:** Grátis, ilimitado, funciona offline
- **Modelo:** Whisper Tiny (faster-whisper)
- **Performance:** 4x mais rápido que openai-whisper
- **Uso de memória:** 60% menor

### Groq Mode (API - Pago)
```bash
AUDIO_TRANSCRIPTION_MODE=groq
GROQ_API_KEY=sua_chave_aqui
```
- **Vantagens:** Mais preciso (Whisper Large v3), muito rápido
- **Modelo:** whisper-large-v3-turbo
- **Limitação:** Arquivos até 25MB (para arquivos maiores, use local)
- **Custo:** Consulte [Groq Pricing](https://groq.com)

## 🔧 Configuração Avançada

### Modelos LLM Disponíveis
Via OpenRouter, você pode usar:
- **Anthropic:** Claude 3.5 Sonnet, Claude 3 Opus
- **OpenAI:** GPT-4o, GPT-4 Turbo
- **Google:** Gemini 2.0 Flash, Gemini Pro
- **Meta:** Llama 3.1 70B
- E muitos mais...

### Ajustando Qualidade vs Velocidade
No arquivo `src/nodes/transcribe_audio.py`, mude o modelo:
```python
# Local (faster-whisper)
model_size = "tiny"  # tiny, base, small, medium, large

# Groq
GROQ_WHISPER_MODEL="whisper-large-v3-turbo"  # ou whisper-large-v3
```

## 📊 API Endpoints

### `POST /upload`
Upload e processamento de vídeo
- **Body:** FormData com `video` e `criteria`
- **Response:** JSON com `session_id`

### `GET /status/<session_id>`
Verifica status do processamento
- **Response:** JSON com `status` (processing/completed)

### `GET /clips/<session_id>`
Lista todos os clips gerados
- **Response:** JSON com array de clips e URLs

### `GET /video/<session_id>/<filename>`
Serve o arquivo de vídeo processado
- **Response:** Video file (MP4)

## 🐛 Troubleshooting

### FFmpeg não encontrado
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Baixe de https://ffmpeg.org/download.html
```

### Erro de memória no modo local
- Use um modelo menor: mude `model_size = "tiny"` no código
- Ou use o modo Groq para processamento em nuvem

### Groq API retorna "Request Entity Too Large"
- Arquivos muito grandes (>25MB) não são suportados pelo Groq
- Solução: Use `AUDIO_TRANSCRIPTION_MODE=local`

### Clips muito curtos ou cortando no meio das frases
- O sistema já tem proteção contra isso
- Se persistir, verifique se o prompt do critério está claro

## 📝 Exemplos de Uso

### Exemplo 1: Extrair citações sobre carreira tech
```
Critério: "Capture partes onde fala sobre carreira tech e dicas para desenvolvedores"
```

### Exemplo 2: Momentos engraçados
```
Critério: "Momentos engraçados, cômicos, que geram risadas"
```

### Exemplo 3: Citações de pessoas específicas
```
Critério: "Partes onde cita Jeff Bezos ou fala sobre Amazon"
```

### Exemplo 4: Conteúdo educacional
```
Critério: "Momentos onde explica conceitos técnicos de programação"
```

## 🚀 Performance

### Tempos de processamento (vídeo de 10 minutos)
- **Transcrição (local):** ~2-3 minutos
- **Transcrição (Groq):** ~30-60 segundos
- **Análise LLM:** ~1-2 minutos
- **Edição de vídeo:** ~30-60 segundos
- **Total:** 4-7 minutos (local) ou 2-4 minutos (Groq)

### Requisitos de sistema
- **CPU:** Recomendado 4+ cores
- **RAM:** 8GB mínimo, 16GB recomendado
- **Disco:** 500MB+ para vídeos temporários

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📞 Suporte

Para dúvidas ou problemas:
- Abra uma issue no GitHub
- Consulte a documentação em `WEB_APP.md`
