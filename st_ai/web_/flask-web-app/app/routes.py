from flask import Blueprint, render_template, request, jsonify, session
import uuid
import os
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    if 'chat_sessions' not in session:
        session['chat_sessions'] = {}
    if 'current_chat' not in session:
        session['current_chat'] = str(uuid.uuid4())
        session['chat_sessions'][session['current_chat']] = []
    return render_template('index.html')

@main.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    chat_id = session.get('current_chat')
    if not chat_id:
        chat_id = str(uuid.uuid4())
        session['current_chat'] = chat_id
        session['chat_sessions'][chat_id] = []
    bot_response = f"Student AI: You said '{user_message}'"
    session['chat_sessions'][chat_id].append({'user': user_message, 'bot': bot_response})
    session.modified = True
    return jsonify({'response': bot_response})

@main.route('/history', methods=['GET'])
def history():
    chat_sessions = session.get('chat_sessions', {})
    return jsonify(chat_sessions)

@main.route('/new_chat', methods=['POST'])
def new_chat():
    chat_id = str(uuid.uuid4())
    session['current_chat'] = chat_id
    session['chat_sessions'][chat_id] = []
    session.modified = True
    return jsonify({'chat_id': chat_id})

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return jsonify({'success': True, 'filename': filename})
    return jsonify({'error': 'Invalid file type'}), 400