"""
Baseline 1: 

Without prediction...
According to the training data to capture the preference list of workers and tasks;
Based on these two preference lists to make the bilateral matching to obtain the stable matching rersult.

Details:
claculate all poi frequency of each worker, if the worker never been to a poi location, the frequency of the poi is zero.

"""


import pandas as pd
import pickle
import numpy as np
from collections import Counter, defaultdict
from utils import top_k_index, dict_len
import copy
import json
import time

from math import sin, asin, cos, radians, fabs, sqrt
EARTH_RADIUS = 6371  # 地球平均半径，6371km

def hav(theta):
    s = sin(theta / 2)
    return s * s

def get_distance_hav(location_a, location_b):

    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(location_a[0])
    lat1 = radians(location_b[0])
    lng0 = radians(location_a[1])
    lng1 = radians(location_b[1])

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance

# 2550 -> 1680 -> 1670 index
def array_pre(pre_w2t_path, pre_t2w_path, test_path, new_test_path, DATASET):

    with open(pre_w2t_path, 'rb') as f_worker, open(pre_t2w_path, 'rb') as f_task:
        worker2task_pre = pickle.load(f_worker)
        task2worker_pre = pickle.load(f_task)
    test_new_df = pd.read_csv(new_test_path)
    
    user_ids = list(set(test_new_df['user_id'].tolist()))
    poi_ids = list(set(test_new_df['poi_id'].tolist()))

    worker_list = []
    for x,_ in worker2task_pre:
        if int(x.split('_')[0]) in user_ids:
            worker_list.append(int(x.split('_')[0]))
    user_id2idx_dict_small = dict(zip(worker_list, range(len(worker_list))))
    task_list = []
    for x,_ in task2worker_pre:
        if int(x.split('_')[0]) in poi_ids:
            task_list.append(int(x.split('_')[0]))
    task_id2idx_dict_small = dict(zip(task_list, range(len(task_list))))
    
    # generate the preference result all valid tasks and workers (1671, 3682)
    test_df = pd.read_csv(test_path)
    user_ids_old = []
    poi_ids_old = []
    for i in list(set(test_df['user_id'].tolist())):
        if i in user_id2idx_dict_small.keys():
            user_ids_old.append(i)
    for j in list(set(test_df['poi_id'].tolist())):
        if j in task_id2idx_dict_small.keys():
            poi_ids_old.append(j)

    small_arr_w2t = np.zeros((len(user_id2idx_dict_small.keys()), len(task_id2idx_dict_small.keys())))
    small_arr_t2w = np.zeros((len(task_id2idx_dict_small.keys()), len(user_id2idx_dict_small.keys())))

    user_id2ids_dict = dict(zip(user_ids, range(len(user_ids))))
    task_id2ids_dict = dict(zip(poi_ids, range(len(poi_ids))))

    for worker_id in user_id2idx_dict_small.keys():
        worker_df = test_new_df[test_new_df['user_id'] == worker_id]
        pre_counter = Counter(worker_df['poi_id'].tolist())
        for t_id in pre_counter.keys():
            task_index = task_id2ids_dict[t_id]
            small_arr_w2t[user_id2ids_dict[worker_id]][task_index] = pre_counter[t_id] / sum(pre_counter.values())

    for task_id in task_id2idx_dict_small.keys():
        task_df = test_new_df[test_new_df['poi_id'] == task_id]
        pre_counter = Counter(task_df['user_id'].tolist())
        for w_id in pre_counter.keys():
            worker_index = user_id2ids_dict[w_id]
            small_arr_t2w[task_id2ids_dict[task_id]][worker_index] = pre_counter[w_id] / sum(pre_counter.values())

    if DATASET == 'Foursquare':
        array_w2t_path = "/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_array_small_w2t_baseline1.csv"
        array_t2w_path = "/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_array_small_t2w_baseline1.csv"
    if DATASET == 'Yelp':
        array_w2t_path = "/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_array_small_w2t_baseline1.csv"
        array_t2w_path = "/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_array_small_t2w_baseline1.csv"

    np.savetxt(array_w2t_path, small_arr_w2t, delimiter=',')
    np.savetxt(array_t2w_path, small_arr_t2w, delimiter=',')

    return array_w2t_path, array_t2w_path, user_id2idx_dict_small, task_id2idx_dict_small


def get_assign_info(user_id2idx_dict_small, task_id2idx_dict_small, new_test_path):
    worker_info = []
    assign_info = []
    test_new_df = pd.read_csv(new_test_path)
    user_ids = list(set(test_new_df['user_id'].tolist()))
    for user_id in set(user_ids):
        user_df = test_new_df[test_new_df['user_id'] == user_id]
        worker_id = user_id2idx_dict_small[user_id]

        if len(user_df) == 1:
            worker_lat = user_df['latitude'].to_list()[-1]
            worker_lon = user_df['longitude'].to_list()[-1]
            worker_starttime = user_df['timestamp'].to_list()[-1]
        else:
            worker_lat = user_df['latitude'].to_list()[-2]
            worker_lon = user_df['longitude'].to_list()[-2]
            worker_starttime = user_df['timestamp'].to_list()[-2]

        task_id = task_id2idx_dict_small[int(user_df['poi_id'].to_list()[-1])]
        task_lat = user_df['latitude'].to_list()[-1]
        task_lon = user_df['longitude'].to_list()[-1]
        task_completetime = user_df['timestamp'].to_list()[-1]
        task_cat = user_df['poi_catid'].to_list()[-1]

        assign_info.append([worker_id, worker_lat, worker_lon, task_id, task_lat, task_lon, worker_starttime, task_completetime, task_cat])
        worker_info.append([worker_id, worker_lat, worker_lon, worker_starttime])
    return assign_info, worker_info


def get_task_info(task_id2idx_dict_small, new_test_path):
    task_info = []
    test_new_df = pd.read_csv(new_test_path)
    task_ids = list(set(test_new_df['poi_id'].tolist()))
    for t_id in task_ids:
        
        task_df = test_new_df[test_new_df['poi_id'] == t_id]
        
        try:
            task_id = task_id2idx_dict_small[t_id]
            # print(task_id)
            task_lat = task_df['latitude'].to_list()[-1]
            task_lon = task_df['longitude'].to_list()[-1]
            task_cat = task_df['poi_catid'].to_list()[-1]
            task_completetime = task_df['timestamp'].to_list()[-1]
            task_info.append([task_id, task_lat, task_lon, task_cat, task_completetime])
        except:
            import pdb; pdb.set_trace()
    
    return task_info

def get_pre_info(array_w2t_path, array_t2w_path, user_id2idx_dict_small, task_id2idx_dict_small, w_range, test_path, new_test_path):
    test_new_df = pd.read_csv(new_test_path)
    user_ids = list(set(test_new_df['user_id'].tolist()))
    poi_ids = list(set(test_new_df['poi_id'].tolist()))
    # generate the preference result all valid tasks and workers (1671, 3682)
    test_df = pd.read_csv(test_path)
    user_ids_old = []
    poi_ids_old = []
    for i in list(set(test_df['user_id'].tolist())):
        if i in user_id2idx_dict_small.keys():
            user_ids_old.append(i)
    for j in list(set(test_df['poi_id'].tolist())):
        if j in task_id2idx_dict_small.keys():
            poi_ids_old.append(j)

    final_user = [user_id2idx_dict_small[k] for k in list(set(user_ids_old).intersection(set(user_ids)))]
    final_poi = [task_id2idx_dict_small[k] for k in list(set(poi_ids_old).intersection(set(poi_ids)))]

    # task info
    import csv
    assign_info, worker_info = get_assign_info(user_id2idx_dict_small, task_id2idx_dict_small, new_test_path)
    task_info = get_task_info(task_id2idx_dict_small, new_test_path)
    if DATASET == 'Foursquare':
        goundtruth_path = 'GroundTruth_data/Foursquare/task_assignment_groundtruth.csv'
        worker_info_path = 'GroundTruth_data/Foursquare/worker_info.csv'
        task_info_path = 'GroundTruth_data/Foursquare/task_info.csv'
    if DATASET == 'Yelp':
        goundtruth_path = 'GroundTruth_data/Yelp_FGRec/task_assignment_groundtruth.csv'
        worker_info_path = 'GroundTruth_data/Yelp_FGRec/worker_info.csv'
        task_info_path = 'GroundTruth_data/Yelp_FGRec/task_info.csv'
    with open(goundtruth_path, 'w') as f_GT:
        writer = csv.writer(f_GT)
        writer.writerow(['worker_id', 'worker_lat', 'worker_lon', 'task_id', 'task_lat', 'task_lon', 'worker_start', 'task_complete', 'task_category'])
        writer.writerows(assign_info)
    with open(worker_info_path, 'w') as f_W:
        writer = csv.writer(f_W)
        writer.writerow(['worker_id', 'worker_lat', 'worker_lon', 'worker_start'])
        writer.writerows(worker_info)
    with open(task_info_path, 'w') as f_T:
        writer = csv.writer(f_T)
        writer.writerow([ 'task_id', 'task_lat', 'task_lon', 'task_category', 'task_complete'])
        writer.writerows(task_info)
    
    # task info & worker info
    if DATASET == 'Foursquare':
        task_df = pd.read_csv('GroundTruth_data/Foursquare/task_info.csv', skiprows=0)
        worker_df = pd.read_csv('GroundTruth_data/Foursquare/worker_info.csv', skiprows=0)
    if DATASET == 'Yelp':
        task_df = pd.read_csv('GroundTruth_data/Yelp_FGRec/task_info.csv', skiprows=0)
        worker_df = pd.read_csv('GroundTruth_data/Yelp_FGRec/worker_info.csv', skiprows=0)
    
    count = -1
    w2t_score_dict = {}
    w2t_df = pd.read_csv(array_w2t_path, header=None)
    for index, row in w2t_df.iterrows():
        line_pre = []
        if index not in final_user:
            continue
        else:
            count += 1
            for p in range(len(row)):
                if p not in final_poi:
                    continue
                else:
                    line_pre.append(row[p])
        w2t_score_dict[index] = line_pre
    w2t_dict = defaultdict(list)
    for i in range(len(w2t_score_dict.keys())):
        worker_i_info = worker_df[worker_df['worker_id']==i]
        worker_lat = worker_i_info['worker_lat']
        worker_lon = worker_i_info['worker_lon']
        worker_i_list = top_k_index(w2t_score_dict[i], [k for k in range(len(w2t_score_dict[i]))], len(w2t_score_dict[i]))
        for task_j in worker_i_list:
            task_j_info = task_df[task_df['task_id']==task_j]
            task_lat = task_j_info['task_lat']
            task_lon = task_j_info['task_lon']
            # import pdb; pdb.set_trace()
            dis = get_distance_hav([worker_lat, worker_lon], [task_lat, task_lon])
            if dis > w_range:
                continue
            w2t_dict[i].append(task_j)
    t2w_score_dict = {}
    t2w_df = pd.read_csv(array_t2w_path, header=None)
    for index, row in t2w_df.iterrows():
        line_pre = []
        if index not in final_poi:
            continue
        else:
            for p in range(len(row)):
                if p not in final_user:
                    continue
                else:
                    line_pre.append(row[p])
        t2w_score_dict[index] = line_pre
    t2w_dict = defaultdict(list)
    for j in range(len(t2w_score_dict.keys())):
        task_j_info = task_df[task_df['task_id']==j]
        task_lat = task_j_info['task_lat']
        task_lon = task_j_info['task_lon']
        task_j_list = top_k_index(t2w_score_dict[j], [k for k in range(len(t2w_score_dict[j]))], len(t2w_score_dict[j]))
        for worker_i in task_j_list:
            worker_i_info = worker_df[worker_df['worker_id']==worker_i]
            worker_lat = worker_i_info['worker_lat']
            worker_lon = worker_i_info['worker_lon']
            dis = get_distance_hav([worker_lat, worker_lon], [task_lat, task_lon])
            if dis > w_range:
                continue
            t2w_dict[j].append(worker_i)

    w_range = str(w_range)
    if DATASET == 'Foursquare':
        w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_baseline1_w2t_order_dict_range%s.json'%w_range
        t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_baseline1_t2w_order_dict_range%s.json'%w_range
        with open(w2t_dict_path, 'w') as f_w2t:  
            json.dump(w2t_dict, f_w2t)
        with open(t2w_dict_path, 'w') as f_t2w:
            json.dump(t2w_dict, f_t2w)
        w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_baseline1_w2t_score_dict_range%s.json'%w_range
        t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_baseline1_t2w_score_dict_range%s.json'%w_range
        with open(w2t_score_dict_path, 'w') as f_w2t:
            json.dump(w2t_score_dict, f_w2t)
        with open(t2w_score_dict_path, 'w') as f_t2w:
            json.dump(t2w_score_dict, f_t2w)
    
    if DATASET == 'Yelp':
        w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_w2t_order_dict_range%s.json'%w_range
        t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_t2w_order_dict_range%s.json'%w_range
        with open(w2t_dict_path, 'w') as f_w2t:  
            json.dump(w2t_dict, f_w2t)
        with open(t2w_dict_path, 'w') as f_t2w:
            json.dump(t2w_dict, f_t2w)     
        w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_w2t_score_dict_range%s.json'%w_range
        t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_t2w_score_dict_range%s.json'%w_range
        with open(w2t_score_dict_path, 'w') as f_w2t:
            json.dump(w2t_score_dict, f_w2t)
        with open(t2w_score_dict_path, 'w') as f_t2w:
            json.dump(t2w_score_dict, f_t2w)
    
    return w2t_dict_path, t2w_dict_path, w2t_score_dict_path, t2w_score_dict_path


def load_data(w2t_dict_path, t2w_dict_path, w2t_score_dict_path, t2w_score_dict_path):

    f_w2t = open(w2t_dict_path)
    w2t_dict_init = json.load(f_w2t)
    f_t2w = open(t2w_dict_path)
    t2w_dict_init = json.load(f_t2w)
    w2t_dict = {}
    t2w_dict = {}
    for w in w2t_dict_init.keys():
        w2t_dict[int(w)] = w2t_dict_init[w]
    for t in t2w_dict_init.keys():
        t2w_dict[int(t)] = t2w_dict_init[t]

    f_score_w2t = open(w2t_score_dict_path)
    f_score_t2w = open(t2w_score_dict_path)
    w2t_score_dict_init = json.load(f_score_w2t)
    t2w_score_dict_init = json.load(f_score_t2w)

    w2t_score_dict = {}
    t2w_score_dict = {}

    for w in w2t_score_dict_init.keys():
        w2t_score_dict[int(w)] = w2t_score_dict_init[w]
    for t in t2w_score_dict_init.keys():
        t2w_score_dict[int(t)] = t2w_score_dict_init[t]
    
    return w2t_dict, t2w_dict, w2t_score_dict, t2w_score_dict


def bi_assign(w2t_dict, t2w_dict, w_capacity, t_capacity):
    count = 0
    assignment_new = defaultdict(list)
    assignment_old = {1}
    worker_assignment = defaultdict(list)

    # Initialize the worker & task dict
    worker_valid_dict = copy.deepcopy(w2t_dict)
    task_valid_dict = defaultdict(list)
    while assignment_old != assignment_new:
        task_temp_dict = defaultdict(list)
        assignment_old = copy.deepcopy(worker_valid_dict)
        # worker selct first
        for worker_id in w2t_dict.keys():
            # global preference score index i
            worker_i_index = w2t_dict[worker_id]
            # candidate task list index
            cd_task_index = copy.deepcopy(worker_valid_dict[worker_id])
            # candidate task list score
            # cd_task_preference = [worker_i_prelist[i] for i in cd_task_index] 
            if w_capacity - len(worker_assignment[worker_id]) > 0:
                fill_tasks_num = w_capacity - len(worker_assignment[worker_id])
                for k in cd_task_index[:fill_tasks_num]:
                    worker_assignment[worker_id].append(k)
                    task_temp_dict[k].append(worker_id)
                    try:
                        worker_valid_dict[worker_id].remove(k)
                    except:
                        import pdb; pdb.set_trace()
            # print(worker_id, len(worker_valid_dict[worker_id]))
        
        # task start select
        for task_id in task_temp_dict.keys():
            # task global preference index
            task_j_index = t2w_dict[task_id]
            # candidate worker list index
            cd_worker_index = task_temp_dict[task_id] + task_valid_dict[task_id]
            # candidate sorted worker list index
            cd_worker_index_global = [task_j_index.index(i) for i in cd_worker_index]
            zip_pairs = sorted(zip(cd_worker_index, cd_worker_index_global), key= lambda x:x[1])
            # top t_capacity worker list
            if t_capacity - len(task_valid_dict[task_id]) > 0:
                top_k_worker = [z[0] for z in zip_pairs][:t_capacity - len(task_valid_dict[task_id])]
            else:
                top_k_worker = [z[0] for z in zip_pairs][:t_capacity]
            # give the temp result to task dict
            task_valid_dict[task_id] = top_k_worker

        # update the selection of workers
        for worker_id in worker_assignment.keys():
            for task_id in worker_assignment[worker_id]:
                if worker_id not in task_valid_dict[task_id]:
                    worker_assignment[worker_id].remove(task_id)
        assignment_new = copy.deepcopy(worker_valid_dict)

        print(dict_len(assignment_old))
        print(dict_len(assignment_new), worker_assignment[0])
        
        count += 1
        print(f'==================== {count} =====================')
    
    return worker_assignment


def success_rate(w2t_score_dict, t2w_score_dict, bi_assign_path, DATASET):
    my_assignment = np.load(bi_assign_path, allow_pickle=True)
    my_assignment = my_assignment.item()

    w2t_allscore = 0
    t2w_allscore = 0
    for w_id in my_assignment.keys():
        # print(w_id, worker_assignment[w_id])
        if len(my_assignment[w_id]) != 0:
            for t_id in my_assignment[w_id]:
                w2t_score = w2t_score_dict[w_id][t_id]
                t2w_score = t2w_score_dict[t_id][w_id]
                w2t_allscore += w2t_score
                t2w_allscore += t2w_score

    if DATASET == 'Foursquare':
        # task info
        task_df = pd.read_csv('GroundTruth_data/Foursquare/task_info.csv', skiprows=0)
        task_dict = defaultdict(dict)
        # worker info
        worker_df = pd.read_csv('GroundTruth_data/Foursquare/worker_info.csv', skiprows=0)
        worker_dict = defaultdict(dict)
        # assign info
        assign_df = pd.read_csv('GroundTruth_data/Foursquare/task_assignment_groundtruth.csv', skiprows=0)
    
    if DATASET == 'Yelp':
        # task info
        task_df = pd.read_csv('GroundTruth_data/Yelp_FGRec/task_info.csv', skiprows=0)
        task_dict = defaultdict(dict)
        # worker info
        worker_df = pd.read_csv('GroundTruth_data/Yelp_FGRec/worker_info.csv', skiprows=0)
        worker_dict = defaultdict(dict)
        # assign info
        assign_df = pd.read_csv('GroundTruth_data/Yelp_FGRec/task_assignment_groundtruth.csv', skiprows=0)

    
    target_assign_dict = {}
    for index, line in task_df.iterrows():
        task_dict[int(line[0])]['task_lat'] = line[1]
        task_dict[int(line[0])]['task_lon'] = line[2]
        task_dict[int(line[0])]['task_cat'] = int(line[3])
        task_dict[int(line[0])]['task_complete'] = int(line[-1])

    for index, line in worker_df.iterrows():
        worker_dict[int(line[0])]['worker_lat'] = line[1]
        worker_dict[int(line[0])]['worker_lon'] = line[1]
        worker_dict[int(line[0])]['worker_time'] = int(line[-1])

    for index, line in assign_df.iterrows():
        target_assign_dict[int(line[0])] = int(line[3])
    
    # calculate the assignment success rate
    """
    if the assigned task has the same category as the true task, thus, the assignment is success.
    """
    success_count = 0
    for worker_id in my_assignment.keys():

        task_list = my_assignment[worker_id]
        if len(task_list) == 0:
            continue
        try:
            task_id_list = task_list
            pred_task_cat = [task_dict[i]['task_cat'] for i in task_id_list]
            # real_task = target_assign_dict[user_id2idx_dict_small[worker_id]]
            real_task = target_assign_dict[worker_id]
            real_task_cat = task_dict[real_task]['task_cat']
            # print(f'pred task: {task_id_list}; real_task: {real_task}; pred cat: {pred_task_cat}; real cat: {real_task_cat}')
        except:
            import pdb; pdb.set_trace()
        if real_task_cat in pred_task_cat[:1]:
            success_count += 1

    whole_assignment = 0
    assigned_worker = 0
    for i in my_assignment.keys():
        if len(my_assignment[i]) != 0:
            assigned_worker += 1
            whole_assignment += len(my_assignment[i])
    print(f'rate: {success_count/assigned_worker}; success_count: {success_count}; assigned_worker: {assigned_worker}')
    # import pdb; pdb.set_trace()
    return success_count/assigned_worker


if __name__ == '__main__':

    # DATASET = 'Foursquare'
    DATASET = 'Yelp'
    w_range = 1   #  (km)
    shuffle = True
    w_capacity = 5
    t_capacity = 1

    baseline = 'WATA'
    # baseline = 'WPTA'
    # baseline = 'WATP'

    print(f'DATASET: {DATASET}; Range: {w_range}')

    print(f'--> preference array')
    if DATASET == 'Foursquare':
        pre_w2t_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursqaure_worker_to_task_preference_time_newcat.pkl'
        pre_t2w_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_task_to_worker_preference_time_newcat.pkl'
        test_path = 'data-process/Foursquare/Foursquare_test.txt'
        new_test_path = 'data-process/Foursquare/Foursquare_test_new.txt'
    if DATASET == 'Yelp':
        pre_w2t_path = 'Preference_value/Yelp_FGRec/Yelp_worker_to_task_preference_time.pkl'
        pre_t2w_path = 'Preference_value/Yelp_FGRec/Yelp_task_to_worker_preference_time.pkl'
        test_path = 'data-process/Yelp_FGRec/Yelp_test.txt'
        new_test_path = 'data-process/Yelp_FGRec/Yelp_test_new.txt'
    # array_w2t_path, array_t2w_path, user_id2idx_dict_small, task_id2idx_dict_small = array_pre(pre_w2t_path, pre_t2w_path, test_path, new_test_path, DATASET)

    print(f'--> w2t & t2w dict path') 
    # w2t_dict_path, t2w_dict_path, w2t_score_dict_path, t2w_score_dict_path = get_pre_info(array_w2t_path, array_t2w_path, user_id2idx_dict_small, task_id2idx_dict_small, w_range, test_path, new_test_path)

    print(f'--> w2t & t2w dict')
    w_range_str = str(w_range)
    if DATASET == 'Foursquare':
        w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_baseline1_w2t_order_dict_range%s.json'%w_range_str
        t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_baseline1_t2w_order_dict_range%s.json'%w_range_str
        w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_baseline1_w2t_score_dict_range%s.json'%w_range_str
        t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_baseline1_t2w_score_dict_range%s.json'%w_range_str
    if DATASET == 'Yelp':
        if baseline == 'WATA':
            w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_w2t_order_dict_range%s.json'%w_range_str
            t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_t2w_order_dict_range%s.json'%w_range_str
            w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_w2t_score_dict_range%s.json'%w_range_str
            t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_t2w_score_dict_range%s.json'%w_range_str
        if baseline == 'WPTA':
            w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_w2t_order_dict_range%s.json'%w_range_str
            t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_t2w_order_dict_range%s.json'%w_range_str
            w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_w2t_score_dict_range%s.json'%w_range_str
            t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_t2w_score_dict_range%s.json'%w_range_str
        if baseline == 'WATP':
            w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_w2t_order_dict_range%s.json'%w_range_str
            t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_t2w_order_dict_range%s.json'%w_range_str
            w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_baseline1_w2t_score_dict_range%s.json'%w_range_str
            t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_t2w_score_dict_range%s.json'%w_range_str
    
    w2t_dict, t2w_dict, w2t_score_dict, t2w_score_dict = load_data(w2t_dict_path, t2w_dict_path, w2t_score_dict_path, t2w_score_dict_path)
     
    if shuffle == True:
        # shuffle the order of task and worker rank
        # for i in w2t_dict.keys():
        #     print(f'task i: {i}: {len(w2t_dict[i])}')
        # import pdb; pdb.set_trace()   

        count_list = []
        for i in w2t_score_dict.keys():
            nonzero_count = 0
            for j in w2t_score_dict[i]:
                if j != 0:
                    nonzero_count += 1

            keep = w2t_dict[i][:nonzero_count]
            left = w2t_dict[i][nonzero_count:len(w2t_dict[i])]
            from random import shuffle
            shuffle(left)
            new_list = keep + left
            count_list.append(nonzero_count)
            w2t_dict[i] = new_list

        for i in t2w_dict.keys():
            
            nonzero_count = 0
            for j in t2w_dict[i]:
                if j != 0:
                    nonzero_count += 1
            # print(nonzero_count)
            # print(f'{i}: {len(t2w_score_dict[i])}')
            try:
                keep = t2w_dict[i][:nonzero_count]
                left = t2w_dict[i][nonzero_count:len(t2w_dict[i])]
            except:
                import pdb; pdb.set_trace()
            from random import shuffle
            shuffle(left)
            new_list = keep + left
            count_list.append(nonzero_count)
            t2w_dict[i] = new_list
        if DATASET == 'Foursquare':
            bi_assign_path = '/nas/project/xieyuan/Bi-preference-TA/Stage2-Assignment/Foursquare/shuffle_baseline_wrange%s_assignment_wcapacity%d_tcapacity%d.npy'%(str(w_range), w_capacity, t_capacity)
        if DATASET == 'Yelp':
            bi_assign_path = '/nas/project/xieyuan/Bi-preference-TA/Stage2-Assignment/Yelp_FGRec/shuffle_baseline_wrange%s_assignment_wcapacity%d_tcapacity%d.npy'%(str(w_range), w_capacity, t_capacity)

        start_time = time.time()
        my_assignment = bi_assign(w2t_dict, t2w_dict, w_capacity=w_capacity, t_capacity=t_capacity)
        np.save(bi_assign_path, my_assignment)
        end_time = time.time()
        print(f'time: {end_time - start_time}')

    if shuffle == False:
        if DATASET == 'Foursquare':
            bi_assign_path = '/nas/project/xieyuan/Bi-preference-TA/Stage2-Assignment/Foursquare/baseline_wrange%s_assignment_wcapacity%d_tcapacity%d.npy'%(str(w_range), w_capacity, t_capacity)
        if DATASET == 'Yelp':
            bi_assign_path = '/nas/project/xieyuan/Bi-preference-TA/Stage2-Assignment/Yelp_FGRec/baseline_wrange%s_assignment_wcapacity%d_tcapacity%d.npy'%(str(w_range), w_capacity, t_capacity)

        import time
        start_time = time.time()
        my_assignment = bi_assign(w2t_dict, t2w_dict, w_capacity=w_capacity, t_capacity=t_capacity)
        np.save(bi_assign_path, my_assignment)
        end_time = time.time()
        print(f'time: {end_time - start_time}')

    print(f'Para. --> DATASET: {DATASET};  baseline: {baseline}; wcapcity: {w_capacity}; tcapacity: {t_capacity}; range: {w_range}')
    success_assign_rate = success_rate(w2t_score_dict, t2w_score_dict, bi_assign_path, DATASET)




    
