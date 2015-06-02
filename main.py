import sys
from similarity import *
def main(key):
    x = data()
    x.output(key)
if __name__ == '__main__':
    for user in sys.argv[1].split(","):
        print('for %d'% user)
        main(user)