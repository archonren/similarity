import argparse
from similarity import *
def main(tag,model_path,user_data_path,patch_path,verbosity):
    x = data(tag,model_path,user_data_path,patch_path,verbosity)
    return x.similar_tag(tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('tags', metavar='key', type=str, nargs='+',
                   help='the tags')
    parser.add_argument('--topn', metavar='topn', type=int,default=5,
                   help='topn')
    parser.add_argument('--model_path', type=str,default='GoogleNews.bin',
                   help='path to word2vec model')
    parser.add_argument('--user_data_path', type =str,default='mefi.json',
                   help='path to user data')
    parser.add_argument('--patch_path', type =str,default=None,
                   help='path to patch data')
    parser.add_argument('--verbosity',default=False,
                   help='show the progress')

    args = parser.parse_args()
    for tag in args.tags:
        print('for %s'% tag)
        most_similar_tag = main(tag,args.model_path,args.user_data_path,args.patch_path,args.verbosity)
        print(most_similar_tag[:args.topn])
