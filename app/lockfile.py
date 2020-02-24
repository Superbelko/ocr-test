import os

class LockDummy:

    def __init__(self, path):
        pass

    def write(self, s):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class LockFile(LockDummy):
    """Simple .lock file mechanism that allows to monitor progress"""

    def __init__(self, path):
        self.path = path
        self.file = open(path, mode='w+')

    def write(self, s):
        self.file.truncate()
        self.file.seek(0)
        self.file.write(s)
        self.file.flush()

    def __exit__(self, *args, **kwargs):
        try:
            self.file.close()
            os.remove(self.path)
        except:
            pass
    
    def __del__(self):
        self.__exit__()


