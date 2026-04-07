import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_engine():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    
    if all([user, password, host, port, db_name]):
        encoded_password = quote_plus(password)
        db_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{db_name}"
        logger.info("[DB] Connecting to PostgreSQL...")
        return create_engine(db_url)
    else:
        # Fallback to local SQLite
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        db_path = os.path.join(base_dir, "data", "quantsim.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        url = f"sqlite:///{db_path}"
        logger.info(f"[DB] PostgreSQL not configured. Using SQLite: {db_path}")
        return create_engine(url)

