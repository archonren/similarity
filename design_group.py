__author__ = 'Archon_ren'
import json,pickle
from gensim.models import Word2Vec
from gensim import matutils
from numpy import float32 as REAL,array,shape,str,zeros
from scivq import *
from scipy.stats.stats import pearsonr

class user(object):
    def __init__(self,user_dict):
        self.id = user_dict['id']
        self.sig_id = user_dict['sig_id']
        self.tags = user_dict['tags_new']
        self.design_group_id =user_dict['design_group_id']

    def show(self):
        print(self.id)
        print(self.sig_id)
        print(self.design_group_id)
        print(self.tags)

    def output(self):
        return str(self.tags)

class users_data(object):
    def __init__(self,k = 20, data_path = 'users.json',model_path = 'GoogleNews.bin'):
        self.data_path = data_path
        self.model_path = model_path
        self.minimium_model = {}
        self.no_match_tag = []
        self.tag_data = []
        self.k = k
        self.vec_dict = {}
        self.abnormal = []
        self.corr_dict = {}
        self.design_group_tag = {}
        self.design_group_sig = {}
        self.no_match = []
        self.out={}
        self.match_count = {}
        self.no_match_group = []

    def load_Data(self):
        with open(self.data_path) as jsonfile:
            self.users_data = json.load(jsonfile)
            for key, value in self.users_data.items():
                self.users_data[key] = user(value)
        jsonfile.close()

    def load_word_to_vec_model(self):
        self.model = Word2Vec.load_word2vec_format(self.model_path, binary=True)

    def union(self,a, b):
        return list(set(a) | set(b))

    def get_tags(self):
        for value in self.users_data.values():
                self.tag_data = self.union(value.tags,self.tag_data)

    def get_design_group_tag(self):
        for value in self.users_data.values():
            if value.design_group_id != '':
                self.design_group_tag['group'+value.design_group_id] = []
                self.design_group_sig['group'+value.design_group_id] = []
        for value in self.users_data.values():
            if value.design_group_id != '':
                self.design_group_tag['group' + value.design_group_id] = self.union(value.tags,self.design_group_tag['group'+value.design_group_id])
                self.design_group_sig['group'+value.design_group_id] = value.sig_id


    def get_minimium_model(self):
        '''
        :return: minimium model: dict contain just the tag in the bank and corresponding 300 dim vector
        '''
        self.load_word_to_vec_model()
        for item in self.tag_data:
            try:
                vec = self.model[item]
                self.minimium_model[item] = vec
            except KeyError:
                try:
                    mean = []
                    if item != '':
                        for word in item.split():
                            mean.append(self.model[word])
                        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
                        self.minimium_model[item] = mean
                except KeyError:
                    self.no_match_tag.append(item)
        if len(self.no_match_tag) !=0:
            print('%d tags not found in word to vec model'% len(self.no_match_tag))

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
            centers,dist = kmeans(vec_array,self.k)
            code,distance = vq(vec_array,centers)
            for i in range(len(code)):
                self.vec_dict[vec_key[i]] = code[i]

    def vote(self):
        if self.vec_dict == {}:
            print('clustering not completed')
            raise AttributeError
        elif type(self.vec_dict) != dict:
            raise TypeError
        else:
            user_item_dict = {}
            x=0
            for key in self.users_data.keys():
                vote_array = zeros((self.k,1))
                for item in self.users_data[key].tags:
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
            design_item_dict = {}
            for key in self.design_group_tag.keys():
                vote_array = zeros((self.k,1))
                for item in self.design_group_tag[key]:
                    if item not in self.no_match_tag:
                        try:
                            vote_array[self.vec_dict[item]] += 1
                        except KeyError:
                            x +=1
                if sum(vote_array) != 0:
                    design_item_dict[key] = vote_array/sum(vote_array)
                else:
                    self.abnormal.append(key)
            self.design_item_dict = design_item_dict

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
            pass

    def match(self,user_id):
        temp_dict = {}
        short_list = []
        try:
            for design_group_id in self.design_item_dict.keys():
                if self.users_data[user_id].sig_id == self.design_group_sig[design_group_id]:
                    temp_dict[design_group_id] = (pearsonr(self.design_item_dict[design_group_id],self.user_item_dict[user_id])[0])
            most_similar = sorted(temp_dict, key=temp_dict.get,reverse=True)
            return most_similar
        except KeyError:
            pass

    def final_suggestion(self):
        self.corr_dict = {}
        for key in self.users_data.keys():
            if self.users_data[key].sig_id != '':
                short_list = self.match(key)
                self.corr_dict[key] = short_list
        for key in self.corr_dict.keys():
            if self.corr_dict[key] ==None:
                self.no_match.append(key)
            elif len(self.corr_dict[key]) == 0:
                self.no_match.append(key)
            else:
                if self.users_data[key].design_group_id != '':
                    try:
                        self.out[key] = self.corr_dict[key][1]
                    except:
                        self.no_match.append(key)
                else:
                    self.out[key] = self.corr_dict[key][0]
        for key in self.design_group_sig.keys():
            if key not in self.out.values():
                self.no_match_group.append(key)
        for key in self.design_item_dict.keys():
            self.match_count[key] = 0
        for item in self.no_match_group:
            self.match_count[item] = 0
        for k in self.out.values():
            self.match_count[k] +=1
        for key in self.no_match:
            sig = self.users_data[key].sig_id
            pool = []
            for k,v in self.design_group_sig.items():
                if v == sig:
                    pool.append([self.match_count[k],k])
            outcome = sorted(pool,reverse=False)
            if len(outcome) != 0:
                self.out[key]=outcome[0][1]
                self.match_count[outcome[0][1]] +=1

    def load_clustering_result(self):
        try:
            with open('class.dat', 'rb') as infile:
                self.vec_dict = pickle.load(infile)
            infile.close()
            self.k = 1000
        except:
            pass
