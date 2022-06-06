
"""
Created on Sun May 22 10:31:03 2022

@author: Cyx
"""

# First is to build the network to get the time seris
# the code referrd to which in  EH2745 GitHub repository L14

import pandapower as pp  # import pandapower
import os
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import copy
import operator



from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from itertools import cycle




def test_net():
    net = pp.create_empty_network()  # create an empty network
    # Since there is no transformer in the system, the nominal voltage level is 110kV
    # According to the document. bus1 is the slack bus, bus2 and bus3 are the Generator bus, 
    # the others are load bus
    # the length of all lines is 10km
    # create bus
    vn_kv = 110

    bus1 = pp.create_bus(net, vn_kv, name="CLARK", type="s")
    bus2 = pp.create_bus(net, vn_kv, name="AMHERST", type="g")
    bus3 = pp.create_bus(net, vn_kv, name="WINLOCK", type="g")
    bus4 = pp.create_bus(net, vn_kv, name="BOWMAN", type="l")
    bus5 = pp.create_bus(net, vn_kv, name="TROY", type="l")
    bus6 = pp.create_bus(net, vn_kv, name="MAPLE", type="l")
    bus7 = pp.create_bus(net, vn_kv, name="GRAND", type="l")
    bus8 = pp.create_bus(net, vn_kv, name="WAUTAGA", type="l")
    bus9 = pp.create_bus(net, vn_kv, name="CROSS", type="l")

    # create lines
    length_km = 10

    line14 = pp.create_line(net, bus1, bus4, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 14")
    line28 = pp.create_line(net, bus2, bus8, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 28")
    line36 = pp.create_line(net, bus3, bus6, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 36")
    line45 = pp.create_line(net, bus4, bus5, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 45")
    line49 = pp.create_line(net, bus4, bus9, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 49")
    line56 = pp.create_line(net, bus5, bus6, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 56")
    line67 = pp.create_line(net, bus6, bus7, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 67")
    line78 = pp.create_line(net, bus7, bus8, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 78")
    line89 = pp.create_line(net, bus8, bus9, length_km, std_type="149-AL1/24-ST1A 110.0", name="Line 89")

    # create generators
    gen1 = pp.create_gen(net, bus1, p_mw=0, slack=True, name="generator 1")
    gen2 = pp.create_sgen(net, bus2, p_mw=163, q_mvar=0, name="generator 2")
    gen3 = pp.create_sgen(net, bus3, p_mw=85, q_mvar=0, name="generator 3")

    # create loads
    load5 = pp.create_load(net, bus5, p_mw=90, q_mvar=30, name="load 5")
    load7 = pp.create_load(net, bus7, p_mw=100, q_mvar=35, name="load 7")
    load9 = pp.create_load(net, bus9, p_mw=125, q_mvar=50, name="load 9")

    return net


# To create the data for different mode.
def create_data_source(net, mode="", n_timesteps=10):
    profiles = pd.DataFrame()
    if mode == "high_load":
        for i in range(len(net.load)):
            profiles["load_p".format(str(i))] = 1.05 * net.load.p_mw[i] + (
                    0.05 * net.load.p_mw[i] * np.random.random(n_timesteps))
            profiles["load_q".format(str(i))] = 1.05 * net.load.q_mvar[i] + (
                    0.05 * net.load.q_mvar[i] * np.random.random(n_timesteps))
    elif mode == "low_load":
        for i in range(len(net.load)):
            profiles["load_p".format(str(i))] = 0.95 * net.load.p_mw[i] - (
                    0.05 * net.load.p_mw[i] * np.random.random(n_timesteps))
            profiles["load_q".format(str(i))] = 0.95 * net.load.q_mvar[i] - (
                    0.05 * net.load.q_mvar[i] * np.random.random(n_timesteps))

    ds = DFData(profiles)

    return profiles, ds


def create_controllers(net, ds):
    for i in range(len(net.load)):
        ConstControl(net, element="load", variable="p_mw", element_index=[i],
                     data_source=ds, profile_name=["load_p".format(str(i))])
        ConstControl(net, element="load", variable="q_mvar", element_index=[i],
                     data_source=ds, profile_name=["load_q".format(str(i))])
    return net


def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls", log_variables=list())
    # according to the file, only bus voltage angles and magnitiudes are needed.
    # these variables are saved to the hard disk after / during the time series loop

    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')

    return ow


# since every functions for build time seris example are shown above, now is to
# describe the all the operating state.

def high_load_state(net, n_timesteps, output_dir):
    # 1. input the test net and choose the correct operating mode
    mode = "high_load"

    # 2. create (random) data source
    profiles, ds = create_data_source(net, mode, n_timesteps)

    # 3. create controllers (to control P values of the high load)
    net = create_controllers(net, ds)

    # figure out the time steps
    time_steps = range(0, n_timesteps)

    # 4. the output writer with the expected results to be stored to files.
    ow = create_output_writer(net, time_steps, output_dir)

    # 5. the main time series function add the parameter to calculate angles
    run_timeseries(net, time_steps, calculate_voltage_angles=True)


def low_load_state(net, n_timesteps, output_dir):
    mode = "low_load"

    profiles, ds = create_data_source(net, mode, n_timesteps)

    create_controllers(net, ds)

    time_steps = range(0, n_timesteps)

    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    run_timeseries(net, time_steps, calculate_voltage_angles=True)


def gen_disconnect_state(net, n_timesteps, mode, output_dir):
    net = net

    net.sgen.at[1, "in_service"] = False

    if mode == "high_load":
        profiles, ds = create_data_source(net, "high_load", n_timesteps)
    elif mode == "low_load":
        profiles, ds = create_data_source(net, "low_load", n_timesteps)

    create_controllers(net, ds)

    time_steps = range(0, n_timesteps)

    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    run_timeseries(net, time_steps, calculate_voltage_angles=True)
    
    net.gen.at[2, "in_service"] = True


def line_disconnect_state(net, n_timesteps, mode, output_dir):
    net = net
    
    net.line.at[5, "in_service"] = False

    if mode == "high_load":
        profiles, ds = create_data_source(net, "high_load", n_timesteps)
    elif mode == "low_load":
        profiles, ds = create_data_source(net, "low_load", n_timesteps)


    create_controllers(net, ds)

    time_steps = range(0, n_timesteps)

    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    run_timeseries(net, time_steps, calculate_voltage_angles=True)
    
    net.line.at[5, "in_service"] = True


# execute the code
output_dir = os.path.join(tempfile.gettempdir(), "time_series_example")
print("Results can be found in your local temp folder: {}".format(output_dir))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

net = test_net()
n_timesteps = 60
# due to different operation mode, different value are store in different files
output_dir_high_load = os.path.join(tempfile.gettempdir(), "time_series_example", "high_load")
print("Results can be found in your local temp folder: {}".format(output_dir_high_load))
if not os.path.exists(output_dir_high_load):
    os.mkdir(output_dir_high_load)
high_load_state(net, n_timesteps, output_dir_high_load)

# refresh the net
net = test_net()
output_dir_low_load = os.path.join(tempfile.gettempdir(), "time_series_example", "low_load")
print("Results can be found in your local temp folder: {}".format(output_dir_low_load))
if not os.path.exists(output_dir_low_load):
    os.mkdir(output_dir_low_load)
low_load_state(net, n_timesteps, output_dir_low_load)

net = test_net()
output_dir_gen_dis_high = os.path.join(tempfile.gettempdir(), "time_series_example", "generator_disconnect_high")
print("Results can be found in your local temp folder: {}".format(output_dir_gen_dis_high))
if not os.path.exists(output_dir_gen_dis_high):
    os.mkdir(output_dir_gen_dis_high)
gen_disconnect_state(net, n_timesteps, "high_load", output_dir_gen_dis_high)

net = test_net()
output_dir_gen_dis_low = os.path.join(tempfile.gettempdir(), "time_series_example", "generator_disconnect_low")
print("Results can be found in your local temp folder: {}".format(output_dir_gen_dis_low))
if not os.path.exists(output_dir_gen_dis_low):
    os.mkdir(output_dir_gen_dis_low)
gen_disconnect_state(net, n_timesteps, "low_load", output_dir_gen_dis_low)

net = test_net()   
output_dir_line_dis_high = os.path.join(tempfile.gettempdir(), "time_series_example", "line_disconnect_high")
print("Results can be found in your local temp folder: {}".format(output_dir_line_dis_high))
if not os.path.exists(output_dir_line_dis_high):
    os.mkdir(output_dir_line_dis_high)
line_disconnect_state(net, n_timesteps, "high_load", output_dir_line_dis_high)

net = test_net()
output_dir_line_dis_low = os.path.join(tempfile.gettempdir(), "time_series_example", "line_disconnect_low")
print("Results can be found in your local temp folder: {}".format(output_dir_line_dis_low))
if not os.path.exists(output_dir_line_dis_low):
    os.mkdir(output_dir_line_dis_low)
line_disconnect_state(net, n_timesteps, "low_load", output_dir_line_dis_low)

# all the time series files has been created.
# Since the voltage and v_angle are seperate in two files, they have to be generate as
# one, then use the KNN or K-means


# To read the data from excel, first is to find the path of voltage
high_load_v_path = os.path.join(output_dir_high_load,"res_bus","vm_pu.xls")

# Then is to read the excel
high_load_v = pd.read_excel(high_load_v_path, index_col=0)

# the same for the voltage angle
high_load_va_path = os.path.join(output_dir_high_load,"res_bus","va_degree.xls")
high_load_va = pd.read_excel(high_load_va_path, index_col=0)

# then generate the features for high load and add a attribute to high_load file
# add the attribute "state" as label which would be used in KNN
high_load_file = pd.concat([high_load_v, high_load_va], axis = 1,ignore_index = True)
high_load_file["state"] = "high_load"


# the other feature are also processed by this method.

#low load
low_load_v_path = os.path.join(output_dir_low_load,"res_bus","vm_pu.xls")
low_load_v = pd.read_excel(low_load_v_path, index_col=0)

low_load_va_path = os.path.join(output_dir_low_load,"res_bus","va_degree.xls")
low_load_va = pd.read_excel(low_load_va_path, index_col=0)

low_load_file = pd.concat([low_load_v, low_load_va], axis = 1,ignore_index = True)
low_load_file["state"] = "low_load"

#generator disconnect at high load
gen_dis_high_v_path = os.path.join(output_dir_gen_dis_high,"res_bus","vm_pu.xls")
gen_dis_high_v = pd.read_excel(gen_dis_high_v_path, index_col=0)

gen_dis_high_va_path = os.path.join(output_dir_low_load,"res_bus","va_degree.xls")
gen_dis_high_va = pd.read_excel(gen_dis_high_va_path, index_col=0)

gen_dis_high_file = pd.concat([gen_dis_high_v, gen_dis_high_va], axis = 1,ignore_index = True)
gen_dis_high_file["state"] = "generator_disconncet_high"

#generator disconnect at low load
gen_dis_low_v_path = os.path.join(output_dir_gen_dis_low,"res_bus","vm_pu.xls")
gen_dis_low_v = pd.read_excel(gen_dis_low_v_path, index_col=0)

gen_dis_low_va_path = os.path.join(output_dir_gen_dis_low,"res_bus","va_degree.xls")
gen_dis_low_va = pd.read_excel(gen_dis_low_va_path, index_col=0)

gen_dis_low_file = pd.concat([gen_dis_low_v, gen_dis_low_va], axis = 1,ignore_index = True)
gen_dis_low_file["state"] = "generator_disconncet_low"

#line disconnect at high load
line_dis_high_v_path = os.path.join(output_dir_line_dis_high,"res_bus","vm_pu.xls")
line_dis_high_v = pd.read_excel(line_dis_high_v_path, index_col=0)

line_dis_high_va_path = os.path.join(output_dir_line_dis_high,"res_bus","va_degree.xls")
line_dis_high_va = pd.read_excel(line_dis_high_va_path, index_col=0)

line_dis_high_file = pd.concat([line_dis_high_v, line_dis_high_va], axis = 1,ignore_index = True)
line_dis_high_file["state"] = "line_disconncet_high"

#line disconnect at low load
line_dis_low_v_path = os.path.join(output_dir_line_dis_low,"res_bus","vm_pu.xls")
line_dis_low_v = pd.read_excel(line_dis_low_v_path, index_col=0)

line_dis_low_va_path = os.path.join(output_dir_line_dis_low,"res_bus","va_degree.xls")
line_dis_low_va = pd.read_excel(line_dis_low_va_path, index_col=0)

line_dis_low_file = pd.concat([line_dis_low_v, line_dis_low_va], axis = 1,ignore_index = True)
line_dis_low_file["state"] = "line_disconncet_low"


# now add all the file together to get the dataset
# the data was built as two group, one is labelled(for knn) and one is unlabelled (for kmeans)
data_label = pd.concat([high_load_file,low_load_file,gen_dis_high_file,
                  gen_dis_low_file,line_dis_high_file,line_dis_low_file], ignore_index = True)
data_unlabel = data_label.drop(['state'], axis = 1)
data_unlabel = data_unlabel.to_numpy()


# In order to normalize the data, we find the max and min values for each of
# the columns in the data
data_max = np.amax(data_unlabel, axis=0)
data_min = np.amin(data_unlabel, axis=0)
data_unlabel_nor = np.zeros((360,18))


# normalize the data
# drop the attribute.
# and for bus1, the voltage and the voltage angle should be 1 and 0
    
for i in range(0,data_unlabel.shape[1]):
    data_unlabel_n = (data_unlabel[:,i] - data_min[i])/(data_max[i] - data_min[i])
    data_unlabel_nor[:,i] = data_unlabel_n
data_unlabel_nor[:,0] = 1
data_unlabel_nor[:,9] = 0

# the clusting does not have to use train set and test set
# the KNN algorithm need to have train set and test set
# here are the two sets for KNN
# using 'train_test_split' to split the dataset to train set and test set.
data_label_nor = data_unlabel_nor.copy()
data_label_nor = pd.DataFrame(data_label_nor)
data_label_nor['state'] = data_label['state'].copy()
data_label_nor_train, data_label_nor_test = train_test_split(data_label_nor, test_size = 0.2)
data_label_nor_train_label = data_label_nor_train["state"].to_numpy()
data_label_nor_test_label = data_label_nor_test["state"].to_numpy()
data_unlabel_nor_train = data_label_nor_train.drop(['state'], axis = 1).to_numpy()
data_unlabel_nor_test = data_label_nor_test.drop(['state'], axis = 1).to_numpy()




# until now, all the data have been processed


# ------------------------------KMEANS--------------------------------------
# build KMEANS Class to realize the clusting method
# n_cluster is the number of cluster
# epsilon is the limitation of distance
# maxstep is the maxium time of iteration
# build a list to store the cluster message
# the 'index' is the point index, the 'label' is list of value
class KMEANS:
    def __init__(self, n_cluster, epsilon=1e-4, maxstep=700):
        self.n_cluster = n_cluster
        self.epsilon = epsilon 
        self.maxstep = maxstep
        self.N = None
        self.centroids = None
        self.cluster = defaultdict(list)

    def init_param(self, data):
        # initial the parameter
        # initial the centroids
        # self.centroids is to store the centroids imformation
        self.N = data.shape[0]
        random_index = np.random.choice(self.N, size=self.n_cluster)
        self.centroids = [data[i] for i in random_index]  
        # iterate over the entire data set, divide the data into different cluster
        for index, p in enumerate(data):
            self.cluster[self.mark(p)].append(index)
        return

    def _cal_dist(self, centroid, p):
        # the equation of the distance between center and data points
        return np.sqrt(np.sum(np.power((centroid - p), 2)))

    def mark(self, p):
        # this method is to calculate the distance between center and data points
        # and then return the index of the minimum one
        dists = []
        for centroid in self.centroids:
            dists.append(self._cal_dist(centroid, p))
        v = np.min(dists)
        return dists.index(v)

    def update_centroid(self, data):
        # update the centroid coordinate
        for label, index in self.cluster.items():
            self.centroids[label] = np.mean(data[index], axis=0)
        return

    def divide(self, data):
        # repeat the clusting processing
        # why here using deepcope is bcs to creat a new one, not only copy the 
        # address where the cluster dict is.
        tmp_cluster = copy.deepcopy(self.cluster)  
        for label, index in tmp_cluster.items():
            for i in index:
                new_label = self.mark(data[i])
                if new_label == label:
                    continue
                else:
                    self.cluster[label].remove(i)
                    self.cluster[new_label].append(i)
        return

    def cal_err(self, data):
        # calculate the mean squared error loss function
        mse = 0
        for label, index in self.cluster.items():
            partial_data = data[index]
            for p in partial_data:
                mse += self._cal_dist(self.centroids[label], p)
        return mse / self.N

    def inp(self, data):
        #This function would inject the dataset to the class
        self.init_param(data)
        step = 0
        while step < self.maxstep:
            step += 1
            self.update_centroid(data)
            self.divide(data)
            err = self.cal_err(data)
            if err < self.epsilon:
                break
        return

def visualize(data, cluster, centroids):
    color = 'bgrym'
    for col, index in zip(cycle(color), cluster.values()):
        partial_data = data[index]
        plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='*', s=100)
    plt.show()
    return


km = KMEANS(6)
km.inp(data_unlabel_nor)
cluster = km.cluster
centroids = np.array(km.centroids)


visualize(data_unlabel_nor, cluster, centroids)


# ------------------------------KNN--------------------------------------
# KNN algorithm
def knn(trainData, testData, labels, k):
    prediction = []
    k = 1
    for i in testData:
        tempData = np.tile(i,(trainData.shape[0],1))
        tempData = cal_dist(tempData,trainData)
        #'argsort' is to sort the data with the index of the point
        sortData = np.argsort(tempData)
        
        count = {}
        for j in range(k):
            vote = labels[sortData[j]]
            count[vote] = count.get(vote,0) + 1
        sortData = sorted(count.items(), key = operator.itemgetter(1), reverse = True)
        pr = sortData[0][0]
        prediction.append(pr)
    return prediction

    
def cal_dist(vectA, vectB):
#the equation of the distance between center and data points
    return (((vectA - vectB) ** 2).sum(axis = 1))**0.5
    

def accuracy(y_true, y_pred):
    count_true = 0
    for i in range(0, len(y_pred)):
        if y_true[i] == y_pred[i]:
            count_true = count_true + 1
    accuracy = count_true / len(y_pred)
    return accuracy

      
prediction = knn(data_unlabel_nor_train,data_unlabel_nor_test,data_label_nor_train_label, 2)


print(accuracy(data_label_nor_test_label, prediction))




    

    




    
    
