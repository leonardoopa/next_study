from fastapi import FastAPI

app = FastAPI(title="rag-assistant", version="0.1.0")



@app.get("/health")
async def health() -> dict:
    # Step 1 (done): process is alive.
    # Step 2 (yours): also check the database, then report the pgvector version.
    #   - app/config.py: Settings(BaseSettings) reading DATABASE_URL from the environment
    #   - lifespan: open an AsyncConnectionPool at startup, close it at shutdown
    #   - query: SELECT extversion FROM pg_extension WHERE extname = 'vector'
    #   - database down -> HTTP 503, not 500






    return {"status": "ok"}
