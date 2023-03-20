import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

def yelp_poi_cat():
    new_poi_cat = []
    poi_list = []
    cat_list = []
    # add poi category
    with open(os.path.join('Yelp_FGRec', 'Yelp'+'_poi_categories_init.txt'), 'rb') as f1:
        poi_cat_dict = defaultdict()
        init_cat_list = []
        for line in f1:
            line = list(map(int, line.decode('utf8').replace('\n', '').split('\t')))
            poi, poi_cat = line[0], line[1]
            init_cat_list.append(poi_cat)
            if poi not in poi_list:
                new_poi_cat.append([poi, poi_cat])
                poi_list.append(poi)
                cat_list.append(poi_cat)
    
    poi_cat_dict = dict(zip(sorted(set(cat_list)), range(max(init_cat_list))))
    final_list = []
    for i in new_poi_cat:
        temp = [i[0], poi_cat_dict[i[1]]]
        final_list.append(temp)

    np.savetxt(os.path.join('Yelp_FGRec', 'Yelp'+'_poi_categories.txt'), final_list, fmt="%s")


def second_data(day, time):
    date_format = "%Y-%m-%d"
    current = datetime.strptime(day, date_format)
    date_format = "%Y/%m/%d"
    bench = datetime.strptime('1970/1/1', date_format)
    time_format = "%H:%M:%S"
    extra_time = datetime.strptime(time, time_format)
    no_days = current - bench
    delta_time_seconds = no_days.days * 24 * 3600 + extra_time.hour * 3600 + extra_time.minute * 60 + extra_time.second
    return delta_time_seconds


def data_split(train_rate, dir_path, DATASET):

    # cut checkin datas and add 
    with open(os.path.join(dir_path, DATASET+'_checkins.txt'), 'rb') as f:
        checkin_list = []
        small_pois = []
        small_user = []
        lines = f.readlines()[1:]
        for line in lines:
            checkin_record = list(map(float, line.decode('utf8').replace('\n', '').split('\t')))
            checkin_record = list(map(int, checkin_record))
            checkin_list.append(checkin_record)
            small_pois.append(checkin_record[1])
            small_user.append(checkin_record[0])
        
        if DATASET == 'Foursquare':
            small_pois = list(sorted(set(small_pois)))[0:10000]
        if DATASET == 'Yelp':
            small_pois = list(sorted(set(small_pois)))[0:5000]
            small_user = list(sorted(set(small_user)))[0:3000]

        final_checkin_list = []
        for rec in checkin_list:
            if rec[1] in small_pois and rec[0] in small_user:
                final_checkin_list.append(rec)
    
    # add poi category
    with open(os.path.join(dir_path, DATASET+'_poi_categories.txt'), 'rb') as f1:
        poi_cat_dict = defaultdict()
        for line in f1:
            if DATASET == 'Yelp':
                line = list(map(int, line.decode('utf8').replace('\n', '').split(' ')))
            if DATASET == 'Foursquare':
                line = list(map(int, line.decode('utf8').replace('\n', '').split('\t')))

            poi, poi_cat = line[0], line[1]
            if poi in small_pois:
                poi_cat_dict[poi] = poi_cat
    
    # add poi coordinate
    with open(os.path.join(dir_path, DATASET+'_poi_coos.txt'), 'rb') as f2:
        poi_coor_dict = defaultdict()
        for line in f2:
            line = list(map(float, line.decode('utf8').replace('\n', '').split('\t')))
            poi, lat, lon = int(line[0]), line[1], line[2]
            if poi in poi_cat_dict.keys():
                poi_coor_dict[poi] = [lat, lon]
    
    # add traj_1
    time_sorted = sorted(final_checkin_list, key = lambda x : x[2])
    train_data_path = os.path.join(dir_path, DATASET+'_train.txt')
    test_data_path = os.path.join(dir_path, DATASET+'_test.txt')
    all_data_path = os.path.join(dir_path, DATASET+'_train&test.txt')

    print(len(time_sorted))

    train_length = len(time_sorted) * train_rate
    train_checkin = []
    test_checkin = []
    all_checkin = []
    count = 0
    std_foursquare = second_data('2012-04-03', '00:00:00') ## the second of the start of a day
    std_yelp = second_data('2004-10-17', '00:00:00')
    
    if DATASET == 'Foursquare':
        std = std_foursquare
    if DATASET == 'Yelp':
        std = std_yelp

    count = 0

    for checkin in time_sorted:
        count += 1

        if DATASET == 'Foursquare':

            if (len(train_checkin) < train_length) and checkin[1] in poi_coor_dict.keys():

                if checkin[2] > std and checkin[2] < std + 60*60*24:
                    std_current = std
                elif checkin[2] > std + 60*60*24:
                    std_current = std + 60*60*24

                norm_time = int((checkin[2] - std_current) / 1800) / 24
                if norm_time >= 1:
                    norm_time -= int(norm_time)
                print(checkin[2], norm_time)
                checkin.append(norm_time)
                record = checkin + [poi_cat_dict[checkin[1]]] + poi_coor_dict[checkin[1]] + [str(checkin[0])+'_1'] + [str(checkin[1])+'_1']
                train_checkin.append(record)
                all_checkin.append(record)
                
            elif checkin[1] in poi_coor_dict.keys():
                if checkin[2] > std and checkin[2] < std + 60*60*24:
                    std_current = std
                elif checkin[2] > std + 60*60*24:
                    std_current = std + 60*60*24
                norm_time = int((checkin[2] - std_current) / 1800) / 24
                if norm_time >= 1:
                    norm_time -= int(norm_time)
                checkin.append(norm_time)

                record = checkin + [poi_cat_dict[checkin[1]]] + poi_coor_dict[checkin[1]] + [str(checkin[0])+'_1'] + [str(checkin[1])+'_1']
                test_checkin.append(record)
                all_checkin.append(record)
        
        if DATASET == 'Yelp':
            
            if (len(train_checkin) < train_length) and checkin[1] in poi_coor_dict.keys():

                if checkin[2] > std and checkin[2] < std + 60*60*24*7:
                    std_current = std
                elif checkin[2] > std + 60*60*24*7:
                    std_current = std + 60*60*24*7

                norm_time = int((checkin[2] - std_current) / 3600) / 24 / 7
                if norm_time >= 1:
                    norm_time -= int(norm_time)
                # print(f'norm time: {checkin[2], norm_time}')
                checkin.append(norm_time)
                record = checkin + [poi_cat_dict[checkin[1]]] + poi_coor_dict[checkin[1]] + [str(checkin[0])+'_1'] + [str(checkin[1])+'_1']
                train_checkin.append(record)
                all_checkin.append(record)
                
            elif checkin[1] in poi_coor_dict.keys():
                if checkin[2] > std and checkin[2] < std + 60*60*24*7:
                    std_current = std
                elif checkin[2] > std + 60*60*24*7:
                    std_current = std + 60*60*24*7
                norm_time = int((checkin[2] - std_current) / 3600) / 24 / 7
                if norm_time >= 1:
                    norm_time -= int(norm_time)
                checkin.append(norm_time)

                record = checkin + [poi_cat_dict[checkin[1]]] + poi_coor_dict[checkin[1]] + [str(checkin[0])+'_1'] + [str(checkin[1])+'_1']
                test_checkin.append(record)
                all_checkin.append(record)
    
    # user_id,node_name/poi_id,timestamp,poi_catid,latitude,longitude
    try:
        np.savetxt(train_data_path, train_checkin, fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s", delimiter=',', header='user_id,poi_id,timestamp,norm_time,poi_catid,latitude,longitude,trajectory_id,worker_trajectory_id')
        np.savetxt(test_data_path, test_checkin, fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s", delimiter=',', header='user_id,poi_id,timestamp,norm_time,poi_catid,latitude,longitude,trajectory_id,worker_trajectory_id')
        np.savetxt(all_data_path, all_checkin, fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s", delimiter=',', header='user_id,poi_id,timestamp,norm_time,poi_catid,latitude,longitude,trajectory_id,worker_trajectory_id')
    except:
        import pdb; pdb.set_trace()
    
    return train_checkin, test_checkin, all_checkin


# reorgniaze the category of the yelp dataset
# yelp_poi_cat()

dir_path = 'Foursquare'
DATASET = 'Foursquare'

# dir_path = 'data-process/Yelp_FGRec'
# DATASET = 'Yelp'
train_checkin, test_checkin, all_checkin = data_split(train_rate=0.8, DATASET=DATASET, dir_path=dir_path)
   
print(train_checkin[0], train_checkin[-1])
print(test_checkin[0], test_checkin[-1])





