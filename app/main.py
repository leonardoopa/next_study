import logging
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException, Request
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings
from app.routers import ask

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


async def _configure(conn) -> None:
    """Runs once per new connection: teach it the `vector` type, leave it idle."""
    await register_vector_async(conn)
    await conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open the pool once. open=False + explicit open() avoids the deprecated
    # "open in constructor" behavior.
    pool = AsyncConnectionPool(
        get_settings().database_url, min_size=1, max_size=5, open=False, configure=_configure
    )
    await pool.open()
    app.state.pool = pool
    try:
        yield
    finally:
        # Shutdown: close connections cleanly.
        await pool.close()


app = FastAPI(title="rag-assistant", version="0.1.0", lifespan=lifespan)
app.include_router(ask.router)


@app.get("/health")
async def health(request: Request) -> dict:
    """Readiness: the API can reach the database. Reports the pgvector version."""
    try:
        # Short timeout: without it the pool waits 30 s before giving up.
        async with request.app.state.pool.connection(timeout=2) as conn:
            cur = await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = await cur.fetchone()
    except (psycopg.Error, TimeoutError) as exc:
        # 503 = dependency down (not a bug in this code, which would be 500).
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ok", "pgvector": row[0] if row else None}
