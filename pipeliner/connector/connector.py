from pipeliner.utils import read_yaml, auto_provided
from .connection import Connection


@auto_provided
class Connector:
    def __init__(self, clusters_path="clusters.yaml"):
        # Read the ssh.yaml config file
        config = read_yaml(clusters_path)

        self.connections = {
            key: Connection(key, value)
            for key, value in config.items()
        }

        for name, connection in self.connections.items():
            setattr(self, name, connection)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for connection in self.connections.values():
            connection.close()

    def __getitem__(self, key):
        return self.connections[key]
