import http.server
import socketserver
import pathlib
import webbrowser

# Nom du fichier HTML du quiz
HTML_FILE = "quiz.html"

# Vérifie présence du fichier
path = pathlib.Path(HTML_FILE)
if not path.exists():
    raise FileNotFoundError(f"Fichier non trouvé: {HTML_FILE}")

# Port utilisé
PORT = 8000

# Handler HTTP standard
handler = http.server.SimpleHTTPRequestHandler

# Lancement serveur
with socketserver.TCPServer(("", PORT), handler) as httpd:
    url = f"http://localhost:{PORT}/{HTML_FILE}"
    print(f"Serveur local actif: {url}")
    
    # Ouvre automatique dans navigateur
    webbrowser.open(url)
    
    httpd.serve_forever()
