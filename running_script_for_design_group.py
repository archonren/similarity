__author__ = 'Archon_ren'
from design_group import *

if __name__ == '__main__':
    x = users_data()
    x.load_Data()
    x.get_design_group_tag()
    x.get_tags()
    x.get_minimium_model()
    #x.clustering()
    x.load_clustering_result()
    x.vote()
    x.final_suggestion()
    print(x.out)
