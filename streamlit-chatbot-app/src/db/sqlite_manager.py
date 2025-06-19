import sqlite3
import os
from pathlib import Path

class SQLiteManager:
    def __init__(self, db_name='chatbot.db'):
        self.db_path = Path(db_name)
        # Create directory structure if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None

    def connect(self):
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise Exception(f"Database connection error: {str(e)}")

    def execute_query(self, query, params=()):
        if self.connection is None:
            self.connect()
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor

    def fetch_results(self, query, params=()):
        if self.connection is None:
            self.connect()
        cursor = self.execute_query(query, params)
        return cursor.fetchall()

    def close(self):
        if self.connection:
            self.connection.close()