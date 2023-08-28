import q4_app
from waitress import serve

serve(q4_app.app, host="localhost", port=3998)
