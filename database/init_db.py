# rag_app/database/init_db.py
from sqlalchemy import create_engine, text
from config.settings import settings

def init_db():
    engine = create_engine(settings.database.postgres_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    engine.dispose()
