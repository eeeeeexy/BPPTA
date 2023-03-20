
def top_k_index(lst, lst_index, k):
    zip_pairs = sorted(zip(lst_index, lst), key= lambda x:x[1], reverse=True)
    # print(zip_pairs)
    return [z[0] for z in zip_pairs][:k]

def dict_len(dict_sample):
    count = 0
    for i in dict_sample.keys():
        count += len(dict_sample[i]) 
    return count


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


# import pandas as pd
# GT_assignment = pd.read_csv('/home/xieyuan/task-assignment_paper_code/Bi-preference-TA/GroundTruth_data/task_assignment_groundtruth.csv', sep= ',')
# Assignment = {}
# dist_list = []
# for i in range(len(GT_assignment)):
#     line = list(GT_assignment.iloc[i])
#     Assignment[i] = line

#     worker_lat = line[1]
#     worker_lon = line[2]
#     task_lat = line[4]
#     task_lon = line[5]

#     dist_list.append(get_distance_hav([worker_lat, worker_lon], [task_lat, task_lon]) * 1000)

# print('Descriptive statistics for labeled test lon',  pd.Series(dist_list).describe(percentiles=[0.05, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 
#                         0.7, 0.8, 0.9, 1]))
