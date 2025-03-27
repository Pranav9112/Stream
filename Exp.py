class FileError(Exception):

    def __init__(self, message = 'Enter valid file(csv/excel)'):
        self.message = message
        super().__init__()

    def __str__(self):
        return self.message
