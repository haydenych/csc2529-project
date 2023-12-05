import os
from datetime import datetime

class Logger:
    def __init__(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.dirname = dirname
        self.filename = os.path.join(
            self.dirname,
            datetime.now().strftime("%d%m%Y_%H%M%S") + ".txt"
        )

        open(self.filename, "w").close()

    def clear(self):
        with open(self.filename, "r+") as f:
            f.truncate(0)

    def log(self, msg):
        with open(self.filename, "a") as f:
            f.write(msg + "\n")
