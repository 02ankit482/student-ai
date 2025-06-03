from flask import Flask
from config import Config
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .rag import rag_bp
    app.register_blueprint(rag_bp)

    return app