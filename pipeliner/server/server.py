import http.server
import socketserver
import threading
import json
import traceback

from pipeliner.utils import green

from .api import Api, get_handlers


class NonBlockingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass


class Server:
    def __init__(self, directory="pipeliner/client/dist", port=8000):
        self.directory = directory
        self.port = port

        api = Api()

        handlers_post = get_handlers(api, "POST")

        class RequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                super().end_headers()

            def do_POST(self):
                for request in handlers_post:
                    if request.__api_path__ == self.path:
                        try:
                            content_length = int(self.headers["Content-Length"])
                            if content_length > 0:
                                post_data = self.rfile.read(content_length)
                                post_data_dict = json.loads(post_data.decode('utf-8'))
                            else:
                                post_data_dict = {}

                            response = request(**post_data_dict)
                            self.send_response(200)
                        except Exception:
                            stack = traceback.format_exc()
                            print(stack)
                            response = {"error": stack}
                            self.send_response(500)

                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())
                        return

                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Not found"}).encode())

        self.server = NonBlockingHTTPServer(("", self.port), RequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        print(green(f"Server started. Open http://localhost:{self.port} to view the client."))
        self.thread.start()

    def stop(self):
        print("Stopping server...")
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
