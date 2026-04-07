"""
Database Initialization Script
Run this to create all tables in the configured database.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.db_models import init_db

if __name__ == "__main__":
    init_db()
