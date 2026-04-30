from k_means import *


data_file = "toy_data/set1a.txt"
#data_file = "toy_data/set2a.txt"
#data_file = "toy_data/set2b.txt"
#data_file = "toy_data/set2c.txt"

K = 2
#initialization = "random"
initialization = "round_robin"


k_means(data_file, K, initialization)
