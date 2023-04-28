#Deploy the model using flask

import os
from subprocess import run, PIPE

#PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.getcwd()


run(['python', PROJECT_ROOT + '/src/app.py'], stdout=PIPE, stderr=PIPE, universal_newlines=True)

if __name__ == '__main__':
    pass

