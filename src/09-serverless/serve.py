import app
from waitress import serve

serve(app.app, host="localhost", port=8990)
