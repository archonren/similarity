from scivq import *
from gensim.models import Word2Vec
from gensim import matutils
from numpy import float32 as REAL,array,sum,zeros
from scipy.stats.stats import pearsonr
import json,pickle
class data(object):

    def __init__(self,k=1000,model_path='GoogleNews.bin',user_data_path='mefi.json',verbosity = False):
        self.tag_data = {}
        self.user_data = {}
        self.k = k
        self.model_path = model_path
        self.user_data_path = user_data_path
        self.model = None
        self.minimium_model = {}
        self.no_match_tag = []
        self.vec_dict = {}
        self.corr_dict = {}
        self.abnormal = []
        self.user_item_dict = {}
        self.verbose = verbosity

    def union(self,a, b):
        return list(set(a) | set(b))

    def tag_bank(self):
        tags = []
        tags_set = []
        i = 0
        temp_data = []
        with open(self.user_data_path) as json_file:
            user_data = json.load(json_file)
        for value in user_data.values():
            tags = self.union(value,tags)
            i += 1
            if int(i/10000)*10000 == i:
                tags_set.append(tags)
                tags = []
        for k in tags_set:
            self.tag_data = self.union(k,temp_data)

    def save_tag_data(self):
        with open('tag_data.dat', 'wb') as outfile:
            pickle.dump(self.tag_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        outfile.close()
        print('tag data saved')

    def load_word_to_vec_model(self):
        self.model = Word2Vec.load_word2vec_format(self.model_path, binary=True)
        if self.verbose:
            print('word to vec model loaded')

    def load_user_data(self):
        with open("mefi.json") as json_file:
            self.user_data = json.load(json_file)
        json_file.close()

    def load_tag_data(self):
        try:
            with open('tag_data.dat', 'rb') as infile:
               self.tag_data = pickle.load(infile)
            infile.close()
            if self.verbose:
                print('tag data loaded')
        except:
            try:
                self.tag_bank()
                self.save_tag_data()
            except:
                print('data not found')


    def get_minimium_model(self):
        '''
        :return: minimium model: dict contain just the tag in the bank and corresponding 300 dim vector
        '''
        if self.model == None:
            self.load_word_to_vec_model()
        for item in self.tag_data:
            try:
                vec = self.model[item]
                self.minimium_model[item] = vec
            except KeyError:
                try:
                    mean = []
                    for word in item.split():
                        mean.append(self.model[word])
                    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
                    self.minimium_model[item] = mean
                except KeyError:
                    self.no_match_tag.append(item)
        if len(self.no_match_tag) !=0:
            print('%d tags not found in word to vec model'% len(self.no_match_tag))

    def save_minimium_model(self):
        with open('minimium.dat', 'wb') as outfile:
            pickle.dump(self.minimium_model, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        outfile.close()
        if self.verbose:
            print('model saved')

    def load_minimium_model(self):
        try:
            with open('minimium.dat', 'rb') as infile:
                self.minimium_model = pickle.load(infile)
            infile.close()
        except:
            self.get_minimium_model()
            self.save_minimium_model()
        if self.verbose:
            print('model loaded')

    def clustering(self):
        '''
        :return: dict that tells which tag belongs to which cluster {key: tag, value: cluster id}
        '''
        if self.minimium_model == {}:
            print('minimium model has not been load')
            raise AttributeError
        elif type(self.minimium_model) != dict:
            raise TypeError
        else:
            vec = []
            vec_key = []
            for key in self.minimium_model.keys():
                vec_key.append(key)
                vec.append(self.minimium_model[key])
            vec_array = array(vec).reshape((-1,300))
            if self.verbose:
                print('kmean start')
            centers,dist = kmeans(vec_array,self.k)
            if self.verbose:
                print('vq start')
            code,distance = vq(vec_array,centers)
            if self.verbose:
                print('vec dict start')
            for i in range(len(code)):
                self.vec_dict[vec_key[i]] = code[i]
            if self.verbose:
                print('vec dict done')

    def save_clustering_result(self):
        with open('class.dat', 'wb') as outfile:
            pickle.dump(self.vec_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        outfile.close()
        if self.verbose ==True:
            print('class saved')

    def load_clustering_result(self):
        try:
            with open('class.dat', 'rb') as infile:
                self.vec_dict = pickle.load(infile)
            infile.close()
        except:
            self.clustering()
            self.save_clustering_result()
        if self.verbose:
            print('class loaded')

    def vote(self):
        if self.verbose:
            print('vote start')
        if self.vec_dict == {}:
            print('clustering not completed')
            raise AttributeError
        elif type(self.vec_dict) != dict:
            raise TypeError
        else:
            user_item_dict = {}
            x=0
            for key in self.user_data.keys():
                vote_array = zeros((self.k,1))
                for item in self.user_data[key]:
                    if item not in self.no_match_tag:
                        try:
                            vote_array[self.vec_dict[item]] += 1
                        except KeyError:
                            x +=1
                if sum(vote_array) != 0:
                    user_item_dict[key] = vote_array/sum(vote_array)
                else:
                    self.abnormal.append(key)
            self.user_item_dict = user_item_dict
            if self.verbose:
                print('complete')

    def save_vote_table(self):
        with open('vote.dat', 'wb') as outfile:
            pickle.dump(self.user_item_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        outfile.close()
        if self.verbose:
            print('vote saved')

    def load_vote_table(self):
        try:
            with open('vote.dat', 'rb') as infile:
                self.user_item_dict = pickle.load(infile)
            infile.close()
        except:
            self.vote()
            self.save_vote_table()
        if self.verbose:
            print('vote loaded')

    def most_similar(self,key1,topN):
        temp_dict = {}
        short_list = []
        try:
            for key2 in self.user_item_dict.keys():
                if key2!=key1:
                    temp_dict[key2] = (pearsonr(self.user_item_dict[key1],self.user_item_dict[key2])[0])
            most_similar = sorted(temp_dict, key=temp_dict.get,reverse=True)
            for i in range(topN):
                short_list.append(most_similar[i])
            self.corr_dict[key1] = short_list
            return short_list
        except KeyError:
            print('key not in dict')

    def output(self,key,topn = 3):
        if self.verbose:
            print('load tag')
        self.load_tag_data()
        if self.verbose:
            print('load user')
        self.load_user_data()
        if self.verbose:
            print('load model')
        self.load_minimium_model()
        if self.verbose:
            print('load class')
        self.load_clustering_result()
        if self.verbose:
            print('load vote')
        self.load_vote_table()
        if self.verbose:
            print('start pearson')
        print(key,self.user_data[key])
        for item in self.most_similar(key,topn):
            print(item,self.user_data[item])
