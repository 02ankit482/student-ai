# Flask Web Application

This is a Flask web application designed to provide a user-friendly interface for students or any person to ask queries from their data by simply putting their data in form of text and videos.

## Project Structure

```
flask-web-app
├── app
│   ├── __init__.py
│   ├── routes.py
│   ├── models.py
│   ├── static
│   │   └── style.css
│   └── templates
│       └── index.html
├── requirements.txt
├── config.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd flask-web-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

Before running the application, you may need to configure the `config.py` file to set your database connection details and secret keys.

## Running the Application

To start the Flask application, run the following command:
```
flask run
```

The application will be accessible at `http://127.0.0.1:5000/`.

## Features

- User authentication
- Student data management
- Responsive design

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.