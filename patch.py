__author__ = 'Archon_ren'
import json
from gensim.models import Word2Vec
from gensim import matutils
from numpy import float32 as REAL,array
class patchdata(object):
    def __init__(self,model_path,patch_patch,verbosity):
        self.patch_path = patch_patch
        self.user_data = {}
        self.tag_data = []
        self.model = None
        self.verbose = verbosity
        self.model_path = model_path
        self.minimium_model = {}
        self.no_match_tag = []

    def union(self,a, b):
        return list(set(a) | set(b))

    def load_patch_data(self):
        with open(self.patch_path) as json_file:
            user_data = json.load(json_file)
        json_file.close()
        self.user_data = user_data

    def get_tags(self):
        tags = []
        tags_set = []
        i = 0
        temp_data = []
        for value in self.user_data.values():
            tags = self.union(value,tags)
            i += 1
            if int(i/10000)*10000 == i:
                tags_set.append(tags)
                tags = []
        self.tag_data = tags
        for k in tags_set:
            self.tag_data = self.union(k,temp_data)

    def load_word_to_vec_model(self):
        self.model = Word2Vec.load_word2vec_format(self.model_path, binary=True)
        if self.verbose:
            print('word to vec model loaded')

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

    def load(self):
        self.load_patch_data()
        self.get_tags()
        self.get_minimium_model()
