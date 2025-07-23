import sys
import os
# Add root directory to sys.path (pizza_monitoring)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import DB_PATH
from detection_service.database import init_db

init_db(DB_PATH)