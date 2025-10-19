# Prime Radiant Embedding Service

FastAPI service that exposes the [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) model through the `llama.cpp` runtime. The server offers an OpenAI-style `/v1/embeddings` endpoint that can be deployed as a Render web service.

- **Runtime:** Python 3.11, FastAPI, llama-cpp-python
- **Model:** `nomic-ai/nomic-embed-text-v1.5` (GGUF, default `Q4_K_M`)
- **Deployment target:** Render.com using the provided `render.yaml`

## Getting Started

### 1. Clone and install
```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # macOS / Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure (optional)
Copy `.env.example` to `.env` and adjust values as needed.

```bash
cp .env.example .env
```

| Variable | Description | Default |
| --- | --- | --- |
| `MODEL_REPO_ID` | Hugging Face repository containing the GGUF model | `nomic-ai/nomic-embed-text-v1.5-GGUF` |
| `MODEL_FILE` | GGUF filename to download | `nomic-embed-text-v1.5.Q4_K_M.gguf` |
| `MODEL_CACHE_DIR` | Local cache directory for the weights | `.models` |
| `THREADS` | Number of CPU threads to use | `4` |
| `CONTEXT_WINDOW` | `n_ctx` value passed to llama.cpp | `8192` |
| `MAX_BATCH_SIZE` | Maximum batch size for inference | `32` |
| `HOST` | Listener interface used when running the app module directly | `0.0.0.0` |
| `PORT` | API port when running via `python -m app.main` | `8000` |

### 3. Run locally
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The first request warms the model by downloading the GGUF file (if missing) and loading it into memory.

### 4. Request embeddings
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
        "input": ["Prime Radiant embeds", "Nomic embeddings powered by llama.cpp"],
        "normalize": true
      }'
```

## API

- `GET /health` – Service heartbeat including the configured model file.
- `POST /v1/embeddings` – Accepts a string or list of strings in the `input` field and returns embedding vectors.

Example response (truncated):
```jsonc
{
  "object": "list",
  "model": "nomic-ai/nomic-embed-text-v1.5:nomic-embed-text-v1.5.Q4_K_M.gguf",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "embedding": [0.0135, -0.0217, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

## Deploying on Render

1. Push this repository to GitHub/GitLab.
2. Create a new **Web Service** on Render and connect the repo.
3. Render will detect the `render.yaml` file and pre-populate settings:
   - Build command: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3.11 (pinned via `runtime.txt`)
4. Ensure the service plan has enough memory/CPU (the `starter` plan in `render.yaml` works for CPU-only inference).
5. Optionally override environment variables (model variant, cache directory, thread count) in the Render dashboard.

On first boot the service downloads the GGUF file from Hugging Face into Render’s persistent `/opt/render/project/.models` directory. Subsequent deploys reuse the cached weights.

## Notes

- The service uses CPU inference by default (`n_gpu_layers=0`). Adjust Render plan accordingly if you need more throughput.
- Larger GGUF quantizations can replace `MODEL_FILE` but require additional memory and startup time.
- Health and readiness checks are provided via `/health`; configure Render’s health checks to point to this endpoint for better monitoring.
