import os
from datetime import datetime

class Logger:
    def __init__(self, dirname, disable=False):
        self.dirname = dirname
        self.filename = os.path.join(
            self.dirname,
            datetime.now().strftime("%d%m%Y_%H%M%S") + ".txt"
        )
        self.disable = disable

        if not self.disable:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            open(self.filename, "w").close()

    def clear(self):
        if not self.disable:
            with open(self.filename, "r+") as f:
                f.truncate(0)

    def log(self, msg=""):
        if not self.disable:
            with open(self.filename, "a") as f:
                f.write(msg + "\n")

    def enable():
        # TODO
        pass

    def disable():
        # TODO
        pass