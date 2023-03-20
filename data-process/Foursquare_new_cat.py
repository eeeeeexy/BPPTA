import pickle, random
import numpy as np
import pandas as pd
import os
from collections import defaultdict

dir_path = 'data-process/Foursquare'
DATASET = 'Foursquare'

with open(os.path.join(dir_path, DATASET+'_poi_categories_init.txt'), 'rb') as f1, \
    open(os.path.join(dir_path, DATASET+'_poi_categories.txt'), 'w') as f2:
    poi_cat_dict = defaultdict()
    new_lines = []
    for line in f1:
        line = list(map(int, line.decode('utf8').replace('\n', '').split('\t')))

        poi, poi_cat = line[0], line[1]

        k = random.randint(poi_cat*20, poi_cat*20+20)

        new_line = str(poi) + '\t' + str(k) + '\n'

        new_lines.append(new_line)

    f2.writelines(new_lines)
