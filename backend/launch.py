import argparse
from threading import Thread
from time import sleep
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def main():
    parser = argparse.ArgumentParser(description="NEURO AI LAB yerel sunucusu")
    parser.add_argument("--no-browser", action="store_true", help="Tarayıcıyı otomatik açma.")
    parser.add_argument("--port", type=int, default=PORT, help="Yerel sunucu bağlantı noktası.")
    options = parser.parse_args()
    if not 1 <= options.port <= 65535:
        parser.error("Bağlantı noktası 1 ile 65535 arasında olmalı.")
    server = uvicorn.Server(uvicorn.Config("backend.main:app", host=HOST, port=options.port))

    def open_when_ready():
        while not server.started and not server.should_exit:
            sleep(0.25)
        if server.started:
            webbrowser.open(f"http://{HOST}:{options.port}/")

    if not options.no_browser:
        Thread(target=open_when_ready, daemon=True).start()
    server.run()


if __name__ == "__main__":
    main()
