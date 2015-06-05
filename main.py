import argparse
from similarity import *
def main(key,k,model_path,user_data_path,verbosity):
    x = data(k,model_path,user_data_path,verbosity)
    x.output(key)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('keys', metavar='N', type=str, nargs='+',
                   help='the users id')
    parser.add_argument('--k', type=int,default=1000,
                   help='number of cluster')
    parser.add_argument('--model_path', type=str,default='GoogleNews.bin',
                   help='path to word2vec model')
    parser.add_argument('--user_data_path', type =str,default='mefi.json',
                   help='path to user data')
    parser.add_argument('--verbosity',default=False,
                   help='show the progress')

    args = parser.parse_args()
    for user in args.keys:
        print('for %s'% user)
        main(user,args.k,args.model_path,args.user_data_path,args.verbosity)
