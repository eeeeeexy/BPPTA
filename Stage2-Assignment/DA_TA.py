import pandas as pd
from collections import defaultdict
from utils import top_k_index, dict_len
import copy, time
import numpy as np
import pickle
import json

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

def get_pre_info(w2t_pre_path, t2w_pre_path, task_info_path, worker_info_path, w_range, DATASET):

    task_df = pd.read_csv(w2t_pre_path, skiprows=0)
    worker_df = pd.read_csv(t2w_pre_path, skiprows=0)
    preference_data_w2t = pd.read_csv(task_info_path, sep= ',', header=None)
    preference_data_t2w = pd.read_csv(worker_info_path, sep= ',', header=None)

    w2t_score_dict = {}
    for i in range(len(preference_data_w2t)):
        w2t_score_dict[i] = list(preference_data_w2t.iloc[i]) 
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
            dis = get_distance_hav([worker_lat, worker_lon], [task_lat, task_lon])
            if dis > w_range:
                continue
            w2t_dict[i].append(task_j) 
    t2w_score_dict = {}
    for i in range(len(preference_data_t2w)):
        t2w_score_dict[i] = list(preference_data_t2w.iloc[i])
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
    
    import json
    w_range = str(w_range)
    if DATASET == 'Foursquare':
        w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_DATA_w2t_order_dict_range%s.json'%w_range
        t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_DATA_t2w_order_dict_range%s.json'%w_range
        with open(w2t_dict_path, 'w') as f_w2t:  
            json.dump(w2t_dict, f_w2t)
        with open(t2w_dict_path, 'w') as f_t2w:
            json.dump(t2w_dict, f_t2w)
        w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_DATA_w2t_score_dict_range%s.json'%w_range
        t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_DATA_t2w_score_dict_range%s.json'%w_range
        with open(w2t_score_dict_path, 'w') as f_w2t:
            json.dump(w2t_score_dict, f_w2t)
        with open(t2w_score_dict_path, 'w') as f_t2w:
            json.dump(t2w_score_dict, f_t2w)
    
    if DATASET == 'Yelp':
        w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_w2t_order_dict_range%s.json'%w_range
        t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_t2w_order_dict_range%s.json'%w_range
        with open(w2t_dict_path, 'w') as f_w2t:  
            json.dump(w2t_dict, f_w2t)
        with open(t2w_dict_path, 'w') as f_t2w:
            json.dump(t2w_dict, f_t2w)     
        w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_w2t_score_dict_range%s.json'%w_range
        t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_t2w_score_dict_range%s.json'%w_range
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
        print(dict_len(assignment_new))
        print(dict_len(assignment_old))
        count += 1
        print(f'==================== {count} =====================')
    
    return worker_assignment


def success_rate(my_assignment_path, new_test_path, DATASET):
    my_assignment = np.load(my_assignment_path, allow_pickle=True)
    my_assignment = my_assignment.item()

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

    if DATASET == 'Yelp':
        success_count = 0
        for worker_id in my_assignment.keys():
            task_list = my_assignment[worker_id]
            if len(task_list) == 0:
                continue
            try:
                task_id_list = [i for i in task_list]
                pred_task_cat = [task_dict[i]['task_cat'] for i in task_id_list]
                real_task = target_assign_dict[worker_id]
                real_task_cat = task_dict[real_task]['task_cat']
                print(f'pred task: {task_id_list}; real_task: {real_task}; pred cat: {pred_task_cat}; real cat: {real_task_cat}')
            except:
                import pdb; pdb.set_trace()
            if real_task_cat in pred_task_cat:
                success_count += 1
        whole_assignment = 0
        assigned_worker = 0
        for i in my_assignment.keys():
            if len(my_assignment[i]) != 0:
                assigned_worker += 1
                whole_assignment += len(my_assignment[i])
        print(f'rate: {success_count/assigned_worker}; success_count: {success_count}; assigned_worker: {assigned_worker}')

    if DATASET == 'Foursquare':
        user_id2idx_dict_small = dict(zip(range(len(target_assign_dict.keys())), list(sorted(target_assign_dict.keys()))))
        success_count = 0
        for worker_id in my_assignment.keys():
            task_list = my_assignment[worker_id]
            if len(task_list) == 0:
                continue
            try:
                task_id_list = [i for i in task_list]
                pred_task_cat = [task_dict[i]['task_cat'] for i in task_id_list]
                real_task = target_assign_dict[user_id2idx_dict_small[worker_id]]  
                real_task_cat = task_dict[real_task]['task_cat']
                # print(f'pred task: {task_id_list}; real_task: {real_task}; pred cat: {pred_task_cat}; real cat: {real_task_cat}')
            except:
                import pdb; pdb.set_trace()
            if real_task_cat in pred_task_cat:
                success_count += 1
        whole_assignment = 0
        assigned_worker = 0
        for i in my_assignment.keys():
            if len(my_assignment[i]) != 0:
                assigned_worker += 1
                whole_assignment += len(my_assignment[i])
        print(f'rate: {success_count/assigned_worker}; success_count: {success_count}; assigned_worker: {assigned_worker}')
    
    return success_count/assigned_worker


if __name__ == '__main__':  

    # DATASET = 'Foursquare'
    DATASET = 'Yelp'
    w_range = 2 #(km)       
    w_capacity = 5
    t_capacity = 1

    # load the w2t & t2w preference
    data_start_time = time.time()
    print(f'--> get preference data')
    if DATASET == 'Foursquare':
        w2t_pre_path = 'GroundTruth_data/Foursquare/task_info.csv'
        t2w_pre_path = 'GroundTruth_data/Foursquare/worker_info.csv'
        task_info_path = '/nas/project/xieyuan/Bi-preference-TA/GroundTruth_data/Foursquare/w2t_preference.csv'
        worker_info_path = '/nas/project/xieyuan/Bi-preference-TA/GroundTruth_data/Foursquare/t2w_preference.csv'
    if DATASET == 'Yelp':
        w2t_pre_path = 'GroundTruth_data/Yelp_FGRec/task_info.csv'
        t2w_pre_path = 'GroundTruth_data/Yelp_FGRec/worker_info.csv'
        task_info_path = '/nas/project/xieyuan/Bi-preference-TA/GroundTruth_data/Yelp_FGRec/w2t_preference.csv'
        worker_info_path = '/nas/project/xieyuan/Bi-preference-TA/GroundTruth_data/Yelp_FGRec/t2w_preference.csv'
    # w2t_dict, t2w_dict, w2t_score_dict, t2w_score_dict = get_pre_info(w2t_pre_path, t2w_pre_path, task_info_path, worker_info_path, w_range, DATASET)
    data_end_time = time.time()
    
    print(f'--> load data')
    w_range_str = str(w_range)
    if DATASET == 'Foursquare':
        w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_DATA_w2t_order_dict_range%s.json'%w_range_str
        t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_DATA_t2w_order_dict_range%s.json'%w_range_str
        w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_DATA_w2t_score_dict_range%s.json'%w_range_str
        t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_DATA_t2w_score_dict_range%s.json'%w_range_str
    if DATASET == 'Yelp':
        w2t_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_w2t_order_dict_range%s.json'%w_range_str
        t2w_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_t2w_order_dict_range%s.json'%w_range_str
        w2t_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_w2t_score_dict_range%s.json'%w_range_str
        t2w_score_dict_path = '/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_DATA_t2w_score_dict_range%s.json'%w_range_str
    w2t_dict, t2w_dict, w2t_score_dict, t2w_score_dict = load_data(w2t_dict_path, t2w_dict_path, w2t_score_dict_path, t2w_score_dict_path)

    if DATASET == 'Foursquare':
        bi_assign_path = '/nas/project/xieyuan/Bi-preference-TA/Stage2-Assignment/Foursquare/DATA_assignment_wrange%s_wcapacity%d_tcapacity%d.npy'%(str(w_range), w_capacity, t_capacity)
        new_test_path = 'data-process/Foursquare/Foursquare_test_new.txt'
    if DATASET == 'Yelp':
        bi_assign_path = '/nas/project/xieyuan/Bi-preference-TA/Stage2-Assignment/Yelp_FGRec/DATA_assignment_wrange%s_wcapacity%d_tcapacity%d.npy'%(str(w_range), w_capacity, t_capacity)
        new_test_path = 'data-process/Yelp_FGRec/Yelp_test_new.txt'
    
    print(f'--> bilteral assignment')
    start_time = time.time()
    my_assignment = bi_assign(w2t_dict, t2w_dict, w_capacity=w_capacity, t_capacity=t_capacity)
    np.save(bi_assign_path, my_assignment)
    end_time = time.time()
    print(f'time: {end_time - start_time}')
    
    # calculate the success assign rate
    print(f'Para. --> DATASET: {DATASET}; wcapacity: {w_capacity}; tcapacity: {t_capacity}; wrange: {w_range}')
    success_assign_rate = success_rate(bi_assign_path, new_test_path, DATASET)

    # import pdb; pdb.set_trace()    


'''
The Deferred Acceptance algorithm without the range consideration.... maybe that is the reason why the code requires a lot of time

'''