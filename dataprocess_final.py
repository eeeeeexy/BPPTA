import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter

def big_array():
    # task index in big array
    nodes_df = pd.read_csv('Foursquare/task_graph_X.csv')
    task_ids = list(set(nodes_df['node_name/poi_id'].tolist()))
    task_id2idx_dict = dict(zip(task_ids, range(len(task_ids))))
    # print(task_id2idx_dict)


    # worker index in big array
    nodes_df = pd.read_csv('Foursquare/worker_graph_X.csv')
    user_ids = list(set(nodes_df['node_name/worker_id'].tolist()))
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))
    # print(user_id2idx_dict)

    # big worker2task preference (2551, 9722)
    for i, w2t_pre in enumerate(worker2task_pre):
        user_id = w2t_pre[0].split('_')[0]
        max_val = np.max(w2t_pre[1:])
        min_val = np.min(w2t_pre[1:])
        w2t_list = w2t_pre[1:][0]
        w2t_list = np.ravel([(x - min_val) / (max_val - min_val) for x in w2t_pre[1:]])

        for j, t2w_pre in enumerate(task2worker_pre):
            task_id = int(t2w_pre[0].split('_')[0])
            big_arr_w2t[user_id2idx_dict[int(user_id)]][int(task_id2idx_dict[int(task_id)])] = w2t_list[task_id2idx_dict[int(task_id)]]

    # big task2worker preference (9722, 2551)
    for i, t2w_pre in enumerate(task2worker_pre):
        task_id = t2w_pre[0].split('_')[0]
        max_val = np.max(t2w_pre[1:])
        min_val = np.min(t2w_pre[1:])
        t2w_list = t2w_pre[1:][0]
        t2w_list = np.ravel([(x - min_val) / (max_val - min_val) for x in t2w_pre[1:]])

        for j, w2t_pre in enumerate(worker2task_pre):
            user_id = int(w2t_pre[0].split('_')[0])
            big_arr_t2w[int(task_id2idx_dict[int(task_id)])][user_id2idx_dict[int(user_id)]] = t2w_list[user_id2idx_dict[int(user_id)]]
    
    np.savetxt("GroundTruth_data/array_big_w2t.csv", big_arr_w2t, delimiter=' ')
    np.savetxt("GroundTruth_data/array_big_t2w.csv", big_arr_t2w, delimiter=' ')

    return big_arr_w2t, big_arr_t2w


def small_array(worker2task_pre, task2worker_pre, DATASET):
    # worker index in small array
    worker_list = [int(x.split('_')[0]) for x,_ in worker2task_pre]
    user_id2idx_dict_small = dict(zip(worker_list, range(len(worker_list))))
    # print(user_id2idx_dict_small)

    # task index in small array
    task_list = [int(x.split('_')[0]) for x,_ in task2worker_pre]
    task_id2idx_dict_small = dict(zip(task_list, range(len(task_list))))

    # small worker2task preference (1681, 3683)
    for i, w2t_pre in enumerate(worker2task_pre):
        user_id = w2t_pre[0].split('_')[0]
        max_val = np.max(w2t_pre[1:])
        min_val = np.min(w2t_pre[1:])
        w2t_list = w2t_pre[1:][0]
        w2t_list = np.ravel([(x - min_val) / (max_val - min_val) for x in w2t_pre[1:]])

        for j, t2w_pre in enumerate(task2worker_pre):
            task_id = int(t2w_pre[0].split('_')[0])
            small_arr_w2t[user_id2idx_dict_small[int(user_id)]][int(task_id2idx_dict_small[int(task_id)])] = float(w2t_list[task_id2idx_dict_small[int(task_id)]])
        
    # big task2worker preference (3683, 1681)
    for i, t2w_pre in enumerate(task2worker_pre):
        task_id = t2w_pre[0].split('_')[0]
        max_val = np.max(t2w_pre[1:])
        min_val = np.min(t2w_pre[1:])
        t2w_list = t2w_pre[1:][0]
        t2w_list = np.ravel([(x - min_val) / (max_val - min_val) for x in t2w_pre[1:]])

        for j, w2t_pre in enumerate(worker2task_pre):
            user_id = int(w2t_pre[0].split('_')[0])
            small_arr_t2w[int(task_id2idx_dict_small[int(task_id)])][user_id2idx_dict_small[int(user_id)]] = float(t2w_list[user_id2idx_dict_small[int(user_id)]])
    
    if DATASET == 'Foursquare':
        np.savetxt('GroundTruth_data/Foursquare/array_small_w2t.csv', small_arr_w2t, delimiter=',')
        np.savetxt('GroundTruth_data/Foursquare/array_small_t2w.csv', small_arr_t2w, delimiter=',')
    if DATASET == 'Yelp':
        np.savetxt('GroundTruth_data/Yelp_FGRec/array_small_w2t.csv', small_arr_w2t, delimiter=',')
        np.savetxt('GroundTruth_data/Yelp_FGRec/array_small_t2w.csv', small_arr_t2w, delimiter=',')
    
    return small_arr_w2t, small_arr_t2w, user_id2idx_dict_small, task_id2idx_dict_small


def new_test_data(user_id2idx_dict_small, task_id2idx_dict_small, test_path, new_test_path):
    test_checkin_new = []
    
    with open(test_path) as f_test:
        for i, line in enumerate(f_test):
            if i == 0:
                continue
            user_id = line.strip().split(',')[0]
            poi_id = line.strip().split(',')[1]
            if int(user_id) in user_id2idx_dict_small.keys() and int(poi_id) in task_id2idx_dict_small.keys():
                test_checkin_new.append(line.strip())

    print(len(test_checkin_new))
    np.savetxt(new_test_path, test_checkin_new, fmt="%s", 
            header='user_id,poi_id,timestamp,norm_time,poi_catid,latitude,longitude,trajectory_id,worker_trajectory_id')


def get_pre_info(user_id2idx_dict_small, task_id2idx_dict_small, test_path, test_new_path, DATASET):
    test_new_df = pd.read_csv(test_path)
    user_ids = list(set(test_new_df['user_id'].tolist()))
    poi_ids = list(set(test_new_df['poi_id'].tolist()))
    # generate the preference result all valid tasks and workers (1671, 3682)
    test_df = pd.read_csv(test_new_path)
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
    
    if DATASET == 'Foursquare':
        w2t_df = pd.read_csv('GroundTruth_data/Foursquare/array_small_w2t.csv', header=None)
        t2w_df = pd.read_csv('GroundTruth_data/Foursquare/array_small_t2w.csv', header=None)
    if DATASET == 'Yelp':
        w2t_df = pd.read_csv('GroundTruth_data/Yelp_FGRec/array_small_w2t.csv', header=None)
        t2w_df = pd.read_csv('GroundTruth_data/Yelp_FGRec/array_small_t2w.csv', header=None)

    count = -1
    w2t_pre = []
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
        w2t_pre.append(line_pre)

    t2w_pre = []
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
        t2w_pre.append(line_pre)

    if DATASET == 'Foursquare':
        np.savetxt('GroundTruth_data/Foursquare/w2t_preference.csv', w2t_pre, delimiter=',', fmt='%s')
        np.savetxt('GroundTruth_data/Foursquare/t2w_preference.csv', t2w_pre, delimiter=',', fmt='%s')
    if DATASET == 'Yelp':
        np.savetxt('GroundTruth_data/Yelp_FGRec/w2t_preference.csv', w2t_pre, delimiter=',', fmt='%s')
        np.savetxt('GroundTruth_data/Yelp_FGRec/t2w_preference.csv', t2w_pre, delimiter=',', fmt='%s')
    

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


if __name__ == '__main__':

    # DATASET = 'Foursquare'
    DATASET = 'Yelp'

    if DATASET == 'Foursquare':
        with open('/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursqaure_worker_to_task_preference_time_newcat.pkl', 'rb') as f_worker,\
             open('/nas/project/xieyuan/Bi-preference-TA/Preference_value/Foursquare/Foursquare_task_to_worker_preference_time_newcat.pkl', 'rb') as f_task:
            worker2task_pre = pickle.load(f_worker)
            task2worker_pre = pickle.load(f_task)

    if DATASET == 'Yelp':
        with open('/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_worker_to_task_preference_time.pkl', 'rb') as f_worker, \
             open('/nas/project/xieyuan/Bi-preference-TA/Preference_value/Yelp_FGRec/Yelp_task_to_worker_preference_time.pkl', 'rb') as f_task:
            worker2task_pre = pickle.load(f_worker)
            task2worker_pre = pickle.load(f_task)

    small_worker_num = len([int(x.split('_')[0]) for x,_ in worker2task_pre])
    small_task_num = len([int(x.split('_')[0]) for x,_ in task2worker_pre])
    big_worker_num = len(task2worker_pre[0][1])
    big_task_num = len(worker2task_pre[0][1])

    print(f'small_worker_num: {small_worker_num}')
    print(f'big_task_num: {big_task_num}')
    print(f'small_task_num: {small_task_num}')
    print(f'big_worker_num: {big_worker_num}')

    small_arr_w2t = np.zeros((small_worker_num, small_task_num))
    small_arr_t2w = np.zeros((small_task_num, small_worker_num))
    big_arr_w2t = np.zeros((big_worker_num, big_task_num))
    big_arr_t2w = np.zeros((big_task_num, big_worker_num))

    print(f'small w2t array shape: {small_arr_w2t.shape}')
    print(f'small t2w array shape: {small_arr_t2w.shape}')
    print(f'big w2t array shape: {big_arr_w2t.shape}')
    print(f'big t2w array shape: {big_arr_t2w.shape}')

    # # extract the preference info
    # big_arr_w2t, big_arr_t2w = big_array()
    print('--> generate small array')
    small_arr_w2t, small_arr_t2w, user_id2idx_dict_small, task_id2idx_dict_small = small_array(worker2task_pre, task2worker_pre, DATASET)

    # # generate checkin new data
    if DATASET == 'Foursquare':
        test_path = 'data-process/Foursquare/Foursquare_test.txt'
        new_test_path = 'data-process/Foursquare/Foursquare_test_new.txt'
    if DATASET == 'Yelp':
        test_path = 'data-process/Yelp_FGRec/Yelp_test.txt'
        new_test_path = 'data-process/Yelp_FGRec/Yelp_test_new.txt'
    
    # generate new test dataset
    # new_test_data(user_id2idx_dict_small, task_id2idx_dict_small, test_path, new_test_path)
    # # get the w2t & t2w preference score
    print('--> preference info')
    get_pre_info(user_id2idx_dict_small, task_id2idx_dict_small, test_path, new_test_path, DATASET)

    # get final assignment info
    print('--> info...')
    assign_info, worker_info = get_assign_info(user_id2idx_dict_small, task_id2idx_dict_small, new_test_path)
    # # get worker & task info
    task_info = get_task_info(task_id2idx_dict_small, new_test_path)

    import pdb; pdb.set_trace()

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

