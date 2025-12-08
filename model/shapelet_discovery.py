import gc
import numpy as np
import util.auto_pisd as auto_pisd
import util.pst_support_method as pstsm
import util.shapelet_support_method as ssm
import time
import multiprocessing
from functools import partial
import random
from collections import defaultdict
import torch
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
class ShapeletDiscover():
    def __init__(self, num_pip=0.4, processes=4, subset_ratio=0.2, r=3):
        self.window_size =[1,5,10,20,30,50,100,200]  
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = None
        self.list_labels = None
        self.processes = processes
        self.subset_ratio = subset_ratio  
        self.r = r  # Value of r (default 3)

    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_shapelet_info(self, number_of_shapelet):
        # if number_of_shapelet < 1:
        number_of_shapelet = int(number_of_shapelet * self.len_of_ts)
        number_of_shapelet = round(number_of_shapelet / len(self.list_labels))
        if number_of_shapelet == 0:
            number_of_shapelet = 1  
        # number_of_shapelet = int(number_of_shapelet)
        list_group_shapelet = pstsm.find_c_shapelet(self.list_group_ppi[0], number_of_shapelet)
        list_group_shapelet = np.asarray(list_group_shapelet)
        list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
        list_shapelet = list_group_shapelet
        for i in range(1, len(self.list_group_ppi)):
            list_ppi = self.list_group_ppi[i]
            list_group_shapelet = pstsm.find_c_shapelet(list_ppi, number_of_shapelet)
            list_group_shapelet = np.asarray(list_group_shapelet)
            list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
            list_shapelet = np.concatenate((list_shapelet,list_group_shapelet),axis=0)

        return list_shapelet


    def find_ppi(self, i, l, subset_indices=None):
        list_result = []
        ts_pos = self.group_train_data_pos[l][i]
        # print(ts_pos)
        ts_idx = i
        # print(ts_idx)
        t1 = self.group_train_data[l][i]
        if subset_indices is not None:
            eval_indices = subset_indices
        else:
            eval_indices = range(len(self.train_data))
        list_result_window = [[j for j in range(len(self.group_train_data_piss[l][i]))] for w in range(len(self.window_size))]
        for k,w in enumerate(self.window_size):
            pdm = {}
            pdm[i * 10000 + i] = np.zeros((0, 0))
            for p in eval_indices:
                t2 = self.train_data[p]
                matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, w)
                pdm[ts_pos * 10000 + p] = matrix_1
            for j in range(len(self.group_train_data_piss[l][i])):
                ts_pis = self.group_train_data_piss[l][i][j]
                ts_ci_pis = self.group_train_data_ci_piss[l][i][j]
                list_dist = [0] * len(self.train_data)  # Initialize with zeros for all time series
                for p in eval_indices:
                    if p == ts_pos:
                        list_dist[p] = 0
                    else:
                        matrix = pdm[ts_pos * 10000 + p]
                        ts_pcs = auto_pisd.pcs_extractor(ts_pis, w, self.len_of_ts)
                        ts_2_ci = self.train_data_ci[p]
                        pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                        dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos[k],
                                                    self.list_end_pos[k], ts_ci_pis, pcs_ci_list)
                        list_dist[p] = dist

                if subset_indices is not None:
                    subset_dist = [list_dist[idx] for idx in subset_indices]
                    subset_labels = [self.train_labels[idx] for idx in subset_indices]
                    ig = ssm.find_best_split_point_and_info_gain(subset_dist, subset_labels, self.list_labels[l])
                else:
                    ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
                ppi = np.asarray([ts_pos, ts_pis[0], ts_pis[1], ig, self.list_labels[l],ts_ci_pis,ts_idx,w])
                list_result_window[k][j] = ppi
        for j in range(len(self.group_train_data_piss[l][i])):
            best_ig_j = -10000000
            best_window_j = -1
            for k in range(len(self.window_size)):

                if list_result_window[k][j][3] > best_ig_j:
                    best_ig_j = (list_result_window[k][j])[3]
                    best_window_j = k
            list_result.append(list_result_window[best_window_j][j])
        return list_result
    def extract_candidate(self, train_data):
        time1 = time.time()
        self.train_data_piss = np.asarray(
            [auto_pisd.auto_piss_extractor(train_data[i], num_pip=self.num_pip) for i in range(len(train_data))], dtype="object")
        ci_return = [auto_pisd.auto_ci_extractor(train_data[i], self.train_data_piss[i]) for i in range(len(train_data))]
        self.train_data_ci = np.asarray([ci_return[i][0] for i in range(len(ci_return))], dtype="object")
        self.train_data_ci_piss = np.asarray([ci_return[i][1] for i in range(len(ci_return))], dtype="object")
        time1 = time.time() - time1
        print("extracting time: %s" % time1)
 

    def get_subset_indices_greedy_k(self, subset_size):
        pick_random_index = random.choice(range(len(self.train_data)))
        center_indices = [pick_random_index]
        
        result_euclidean = {}  

        while len(center_indices) < subset_size:
            max_dist = -1
            next_index = -1
            for i in range(len(self.train_data)):
                if i in center_indices:
                    continue
                min_dist = float('inf')
                for c_i in center_indices:
                    i1, i2 = min(i, c_i), max(i, c_i)
                    if (i1, i2) not in result_euclidean:
                        dist = euclidean(self.train_data[i1], self.train_data[i2])
                        result_euclidean[(i1, i2)] = dist
                    else:
                        dist = result_euclidean[(i1, i2)]
                    if dist < min_dist:
                        min_dist = dist
                if min_dist > max_dist:
                    max_dist = min_dist
                    next_index = i
            center_indices.append(next_index)
        return center_indices
    def discovery(self, train_data, train_labels, g):
        """
        Two-phase shapelet discovery process
        Parameters:
        train_data: list of time series data
        train_labels: list of corresponding labels
        g: number of final shapelets to select
        r: expansion factor for round 1 selection (default to self.r if not provided)
        """
        # print("len data"+str(len(train_data)))
        # self.window_size = int(self.window_size)
        
        time_start = time.time()
        self.train_data = train_data
        self.train_labels = train_labels

        self.len_of_ts = len(train_data[0])
        self.list_labels = np.unique(train_labels)

        self.window_size = [i for i in self.window_size if i < self.len_of_ts]
        # print(self.window_size)        
        self.list_start_pos = [np.ones(self.len_of_ts, dtype=int) for i in range(len(self.window_size))]
        self.list_end_pos = [np.ones(self.len_of_ts, dtype=int) * (i * 2 + 1) for i in self.window_size]
        for j,w in enumerate(self.window_size) :
            for i in range(w):
                # print(j,w,i)
                self.list_end_pos[j][-(i + 1)] -= w - i
            for i in range(w - 1):
                self.list_start_pos[j][i] += w - i - 1
        num_candidate = sum(len(i) for i in self.train_data_ci)
        # print(f"num candidate: {num_candidate}")

        # Divide time series into group of label
        group_train_data = [[] for i in self.list_labels]
        group_train_data_pos = [[] for i in self.list_labels]
        group_train_data_piss = [[] for i in self.list_labels]
        group_train_data_ci_piss = [[] for i in self.list_labels]

        for l in range(len(self.list_labels)):
            for i in range(len(train_data)):
                if train_labels[i] == self.list_labels[l]:
                    group_train_data[l].append(train_data[i])
                    group_train_data_pos[l].append(i)
                    group_train_data_piss[l].append(self.train_data_piss[i])
                    group_train_data_ci_piss[l].append(self.train_data_ci_piss[i])
                    # print(group_train_data[l])
                    # print(i)
        # return
        self.group_train_data = np.asarray(group_train_data, dtype="object")
        self.group_train_data_pos = np.asarray(group_train_data_pos, dtype="object")
        self.group_train_data_piss = np.asarray(group_train_data_piss, dtype="object")
        self.group_train_data_ci_piss = np.asarray(group_train_data_ci_piss, dtype="object")

        # PHASE 1: Select candidates on a subset of the data
        print("Phase 1: Initial candidate evaluation on subset")
        subset_size = int(len(train_data) * self.subset_ratio)

        subset_indices = self.get_subset_indices_greedy_k(subset_size)
        # print(subset_indices)
       
        round1_group_ppi = []
        for l in range(len(self.list_labels)):
            p = multiprocessing.Pool(processes=self.processes)
            temp_ppi = p.map(partial(self.find_ppi, l=l, subset_indices=subset_indices), range(len(self.group_train_data[l])))
            p.close()
            p.join()
            # return 0
            list_ppi = []
            for i in range(len(self.group_train_data[l])):
                pii_in_i = temp_ppi[i]
                for j in range(len(pii_in_i)):
                    list_ppi.append(pii_in_i[j])
            list_ppi = np.asarray(list_ppi)
            round1_group_ppi.append(list_ppi)
            del temp_ppi
            del list_ppi
            gc.collect()
        number_of_shapelet_per_class = -1
       
        number_of_shapelet_per_class = int(g * self.len_of_ts)
        number_of_shapelet_per_class = round(number_of_shapelet_per_class / len(self.list_labels))
        if number_of_shapelet_per_class == 0:
            number_of_shapelet_per_class = 1

        num_candidate_select = 0


        label_candidates_groups_by_ts_pos = [] 

        for l, label in enumerate(self.list_labels):
            sorted_candidates = round1_group_ppi[l][round1_group_ppi[l][:, 3].argsort()[::-1]]
            top_candidates = sorted_candidates[:number_of_shapelet_per_class * self.r]
            ts_pos_dict = {}
            for c in top_candidates:
                ts_pos = int(c[0])
                if ts_pos not in ts_pos_dict:
                    ts_pos_dict[ts_pos] = []
                ts_pos_dict[ts_pos].append(c)

            grouped_by_ts_pos = list(ts_pos_dict.values())
            for group in grouped_by_ts_pos:
                num_candidate_select += len(group)
            label_candidates_groups_by_ts_pos.append((int(label), grouped_by_ts_pos))

        # print(time.time() - time_distance)
        print(f"Phase 2: Re-evaluating {num_candidate_select} candidates on full dataset")
        # print("evaluate full set")
        time_evaluate = time.time()
        # self.list_group_ppi = [[] for _ in range(len(self.list_labels))]
        # temp_group_ppi = [[] for _ in range(len(self.list_labels))]
        self.list_group_ppi = []
        for label, grouped_by_ts_pos in label_candidates_groups_by_ts_pos:
            p = multiprocessing.Pool(processes=self.processes)
            temp_ppi = p.map(partial(self.evaluate_group_wrapper_ts_pos, label), grouped_by_ts_pos)
            p.close()
            p.join()

            list_ppi = []
            for group_result in temp_ppi:
                list_ppi.extend(group_result)

            self.list_group_ppi.append(np.asarray(list_ppi))
            del temp_ppi, list_ppi
            gc.collect()
       
        # print(time.time() - time_evaluate)
        
        time_total = time.time() - time_start
        print(f"window_size: {self.window_size} - total time: {time_total}")
        return time_total,num_candidate,num_candidate_select
    def evaluate_group_wrapper_ts_pos(self, label,candidates):
            
        num_train_samples = len(self.train_data)
        results = []
        
        i = int(candidates[0][6])
        ts_pos = int(candidates[0][0])
        list_result_window = [[c for c in range(len(candidates))] for w in range(len(self.window_size))]

        for k,w in enumerate(self.window_size):
            pdm = {}
            pdm[i * 10000 + i] = np.zeros((0, 0))
            t1 = self.group_train_data[label][i]
            for p in range(num_train_samples):
                t2 = self.train_data[p]
                matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, w)
                pdm[ts_pos * 10000 + p] = matrix_1
            for j,candidate in enumerate(candidates):
                # pdm = {}    
                start_pos = int(candidate[1])
                end_pos = int(candidate[2])
                ts_ci_pis = candidate[5]
                ts_pis = [start_pos, end_pos]
              
                ts_pcs = auto_pisd.pcs_extractor(ts_pis, w, self.len_of_ts)
              
                list_dist = np.zeros(num_train_samples)
                for p in range(num_train_samples):
                    if p == ts_pos:
                        continue  
                    t2 = self.train_data[p]
                    matrix = pdm[ts_pos * 10000 + p]
                    ts_2_ci = self.train_data_ci[p]
                    pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                    dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos[k],
                                                    self.list_end_pos[k], ts_ci_pis, pcs_ci_list)
                
                    list_dist[p] = dist
                ig = ssm.find_best_split_point_and_info_gain(
                    list_dist, self.train_labels, self.list_labels[label]
                )
                list_result_window[k][j] = np.array([ts_pos, start_pos, end_pos, ig, label,w])

        for j in range(len(candidates)):
            best_ig_j = -10000000
            best_window_j = -1
            for k in range(len(self.window_size)):
                if list_result_window[k][j][3] > best_ig_j:
                    best_ig_j = list_result_window[k][j][3]
                    best_window_j = k
            results.append(list_result_window[best_window_j][j])
        return results

