import os

from pipeliner.utils import read_yaml, auto_provided, orange

from .cluster import Cluster


@auto_provided
class Connector:
    def __init__(self, clusters_path="clusters.yaml"):
        # Read the ssh.yaml config file
        config = read_yaml(clusters_path)

        self.connections = {
            key: Cluster(key, value)
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

    def __iter__(self):
        yield from self.connections.values()

    def log(self, *args, **kwargs):
        print(orange(f"[local]"), *args, **kwargs)

    def sync(self, cluster=None, commit_message=None, debug=False):
        if cluster:
            try:
                self[cluster].git_sync(debug=debug)
                return { "success": True, cluster: dict(success=True) }
            except Exception as e:
                return { "success": False, cluster: dict(success=False, message=str(e)) }

        # If cluster is not specified, then sync everything.

        # First of all, add and push local changes
        self.log(f"Syncing git repo")
        os.system("git add .")
        if os.system("git diff-index --quiet HEAD --") != 0:
            if commit_message is None:
                commit_message = "(automatic commit)"

            if os.system(f"git commit -m \"{commit_message}\"") != 0:
                return dict(success=False, message="Failed to commit changes")

            if os.system("git push") != 0:
                return dict(success=False, message="Failed to push changes")

        # Now, update all the clusters
        statuses = {}
        success = True
        for cluster in self:
            try:
                cluster.git_sync(debug=debug)
                statuses[cluster.name] = dict(success=True)
            except Exception as e:
                success = False
                statuses[cluster.name] = dict(success=False, message=str(e))

        statuses["success"] = success
        return statuses
