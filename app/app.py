from flask import Flask
from app.services.routes import api

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api, url_prefix="/api")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

@app.route("/")
def index():
    return {
        "message": "Echonote backend is running",
        "docs": "/api/health"
    }
