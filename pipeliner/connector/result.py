from dataclasses import dataclass


@dataclass
class Result:
    success: bool
    stdout: str
    stderr: str
    exit_code: int

    def __iter__(self):
        yield self.success
        yield self.stdout
        yield self.stderr
        yield self.exit_code

