# DocuMind Docker Compose deployment

This stack runs PostgreSQL + pgvector, FastAPI, and Streamlit in Docker. Ollama
and Qwen3 remain on the host machine; the API container connects to them through
`host.docker.internal`.

## Files in this bundle

- `Dockerfile.api`: FastAPI image
- `Dockerfile.frontend`: Streamlit image
- `docker-compose.yml`: PostgreSQL, API, and frontend services
- `.env.example`: local Compose configuration template
- `.dockerignore`: avoids copying local data and secrets into build contexts
- `processor_llm_env_patch.txt`: required environment configuration change

## Setup

1. Copy these files into the DocuMind project root.
2. Copy `.env.example` to `.env` and replace the example PostgreSQL password.
3. Apply the `processor/llm.py` patch described in `processor_llm_env_patch.txt`.
4. Ensure Ollama is running on the host and the configured model is available:

   ```powershell
   ollama list
   ollama pull qwen3:4b
   ```

5. Start the stack:

   ```powershell
   docker compose up --build -d
   ```

6. Check service health:

   ```powershell
   docker compose ps
   docker compose logs --tail=100 api frontend
   ```

7. Open:
   - Streamlit: http://localhost:8501
   - FastAPI Swagger: http://localhost:8000/docs
   - API health: http://localhost:8000/health

## Ollama connectivity on Windows

The API container uses `http://host.docker.internal:11434` by default. If the
container cannot reach Ollama, configure Ollama to listen on an address reachable
from Docker Desktop, then restart Ollama. Binding Ollama to `0.0.0.0` can expose
its API beyond the local machine depending on firewall/network settings; restrict
access with Windows Firewall and do not expose port 11434 publicly.

You can override the URL in `.env`:

```dotenv
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:4b
```

## Database initialization and existing data

The SQL migration is mounted into PostgreSQL's initialization directory. The
official PostgreSQL entrypoint runs it only when the database data directory is
initialized for the first time. Your existing named volume is retained when you
run `docker compose down` (without `-v`). Do not run `docker compose down -v` on
your existing setup unless you intentionally want to delete the database volume.

For an existing database, the schema should already exist from the earlier
migration. Verify it before proceeding:

```powershell
docker compose exec postgres psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB -c "SELECT version FROM schema_migrations;"
```

If PowerShell does not have those variables set, use the database/user values
from `.env` in the command.

## Notes

- Ports are bound to `127.0.0.1` so the services are local-only by default.
- The Sentence Transformers model cache is stored in the named
  `documind_model_cache` volume; first use downloads the model inside the API
  container and later runs reuse it.
- PostgreSQL data persists in `documind_postgres_data`.
- The frontend calls the API using Docker's internal service DNS name `api`, not
  `localhost`.
- This is a reproducible local Docker Compose deployment, not a hardened
  public-cloud production deployment. Add TLS, authentication, network controls,
  secret management, and an appropriate reverse proxy before exposing it.
