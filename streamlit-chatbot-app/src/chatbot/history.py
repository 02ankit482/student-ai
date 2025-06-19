from pathlib import Path
from db.sqlite_manager import SQLiteManager

class HistoryManager:
    def __init__(self):
        # Create data directory in project root
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / "data" / "chathistory.db"
        
        # Ensure data directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_manager = SQLiteManager(str(db_path))
        self._initialize_db()

    def _initialize_db(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.db_manager.execute_query(create_table_query)

    def add_message(self, user_id: str, role: str, content: str):
        query = "INSERT INTO chat_messages (user_id, role, content) VALUES (?, ?, ?)"
        self.db_manager.execute_query(query, (user_id, role, content))

    def get_chat_history(self, user_id: str):
        query = """
        SELECT role, content 
        FROM chat_messages 
        WHERE user_id = ? 
        ORDER BY timestamp
        """
        return self.db_manager.fetch_results(query, (user_id))