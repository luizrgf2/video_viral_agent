# Video Viral Agent - Web Interface

Interface web para extração automática de momentos virais de vídeos usando IA.

## 🚀 Como usar

### 1. Inicie o servidor Flask:
```bash
uv run python app.py
```

O servidor estará disponível em: **http://localhost:5000**

### 2. Acesse a interface:
Abra seu navegador em `http://localhost:5000`

### 3. Faça upload do seu vídeo:
- Clique na área de upload ou arraste o arquivo
- Formatos aceitos: MP4, MOV, AVI, MKV
- Tamanho máximo: 500MB

### 4. Descreva o critério:
No campo "Critério de análise", descreva quais momentos você quer extrair.

**Exemplos:**
- "Capture partes onde ele cita desenvolvedores plenos"
- "Momentos engraçados e cômicos"
- "Partes onde fala sobre carreira tech"
- "Citações sobre Amazon e CEO Jeff Bezos"

### 5. Processe:
Clique em "🚀 Processar Vídeo" e aguarde.

### 6. Assista os resultados:
Quando o processamento terminar, você poderá:
- Assistir aos momentos identificados
- Fazer download dos clips individuais
- Todos os clips estarão na pasta `output_clips/`

## 📂 Estrutura de arquivos

```
video_viral_agent/
├── app.py                    # Servidor Flask
├── templates/
│   └── index.html           # Interface web
├── uploads/                  # Vídeos enviados (temporário)
├── output_clips/             # Clips processados (organizados por sessão)
└── src/                      # Lógica de processamento
```

## 🔧 Configuração

### Variáveis de ambiente (.env):
```env
OPENROUTER_API_KEY=sua_chave_aqui
VLM_MODEL_NAME=anthropic/claude-3.5-sonnet
LLM_MODEL_NAME=openai/gpt-4o
```

### Configurações do app (em app.py):
- `MAX_CONTENT_LENGTH`: 500MB (tamanho máximo do vídeo)
- `ALLOWED_EXTENSIONS`: MP4, MOV, AVI, MKV

## 🎯 Funcionalidades

✅ Upload de vídeos (drag & drop ou seleção)
✅ Preview do vídeo antes de processar
✅ Processing em tempo real (async)
✅ Player de vídeo HTML5
✅ Download dos clips processados
✅ Organização por sessão (timestamp)
✅ Verificação de status em tempo real

## 📝 API Endpoints

### POST /upload
Upload e processamento de vídeo

### GET /status/<session_id>
Verifica status do processamento

### GET /clips/<session_id>
Lista todos os clips gerados

### GET /video/<session_id>/<filename>
Serve o arquivo de vídeo

## 🎨 Interface

- Design moderno com Bootstrap 5
- Responsivo (funciona em mobile)
- Feedback visual em tempo real
- Player de vídeo integrado
- Download fácil dos clips

## ⚡ Workflow

1. Upload → 2. Transcrição (Whisper) → 3. Identificação (LLM) → 4. Refinamento → 5. Edição → 6. Resultados

Tempo estimado: 1-5 minutos dependendo do tamanho do vídeo.
