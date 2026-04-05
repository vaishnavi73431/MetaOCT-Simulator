"""
MetaOCT Server - OpenEnv Compatible REST API
Required by openenv validate for multi-mode deployment.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import OpenEnvHandler, get_env
from http.server import HTTPServer
import threading

def main():
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    server = HTTPServer(("0.0.0.0", port), OpenEnvHandler)
    print(f"[INFO] MetaOCT OpenEnv Server running on port {port}", flush=True)
    
    def prewarm():
        get_env("easy")
        print("[INFO] Environment ready.", flush=True)
    
    t = threading.Thread(target=prewarm, daemon=True)
    t.start()
    
    server.serve_forever()

if __name__ == "__main__":
    main()
