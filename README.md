# AI Agent OOP

Multi-agent system built with Django + Ollama.

## Agents
- **ChatBot** - General conversation with keyboard layout detection
- **CodeBot** - Programming & code execution
- **DataBot** - Data analysis (CSV, JSON files)
- **ResearchBot** - Web search & summarization
- **PlannerBot** - Multi-step task planning across agents

## One-Click Deploy

| Platform | Button |
|----------|--------|
| Render | [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mohamed-elmoge/ai_serve) |
| Koyeb | [![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?name=ai-agent-oop&type=git&repository=mohamed-elmoge/ai_serve&branch=master&builder=docker) |

**⚠️ Important:** The app needs an Ollama instance to function. After deploying:
1. Set the `OLLAMA_BASE_URL` environment variable to your Ollama server URL
2. Set `OLLAMA_MODEL` (default: `qwen2.5-coder:14b`)
3. Set `DJANGO_ALLOWED_HOSTS` to include your deployment domain

## Local Development

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Make sure Ollama is running at `http://localhost:11434` with `qwen2.5-coder:14b`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen2.5-coder:14b` | LLM model to use |
| `DJANGO_SECRET_KEY` | (random) | Django secret key |
| `DJANGO_DEBUG` | `True` | Debug mode |
| `DJANGO_ALLOWED_HOSTS` | `192.168.1.5,localhost,127.0.0.1` | Allowed hosts |
