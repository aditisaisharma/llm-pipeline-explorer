#!/usr/bin/env python3
"""
MorphoLLM: Local Development Server
Launches a lightweight HTTP server to explore the LLM pipeline visualizer locally.
"""

import http.server
import socketserver
import os
import sys
import webbrowser

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable caching-free local development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 60)
        print("🦋 MorphoLLM: Ethereal LLM Pipeline Explorer")
        print(f"✨ Running at: {url}")
        print("   Press Ctrl+C to stop.")
        print("=" * 60)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server. Goodbye!")
            httpd.server_close()

if __name__ == "__main__":
    main()
