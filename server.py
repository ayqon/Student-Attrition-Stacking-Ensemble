import os
import http.server
import socketserver

PORT = int(os.environ.get("PORT", 8080))
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    print(f"Serving OULAD Analytics on port {PORT}...")
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Live at http://0.0.0.0:{PORT}/")
        httpd.serve_forever()
