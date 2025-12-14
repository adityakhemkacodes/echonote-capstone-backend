# app/app.py

from flask import Flask
from flask_cors import CORS

from app.services.routes import api

def create_app():
    app = Flask(__name__)

    # ✅ CORS CONFIG (safe for local dev)
    CORS(
        app,
        resources={r"/api/*": {"origins": ["http://localhost:5173"]}},
        supports_credentials=True,
    )

    app.register_blueprint(api, url_prefix="/api")

    @app.route("/")
    def root():
        return {"status": "backend running"}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
