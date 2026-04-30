#Preston Loera, 1001889535
import math
import random

class DataPoint:
    def __init__(self, point, init_cluster_assignment):
        self.point = point
        self.cluster_assigned = init_cluster_assignment

    def updateAssignedCluster(self, new_cluster_id):
        if(new_cluster_id == self.cluster_assigned):
            self.cluster_id_updated = False
        else:
            self.cluster_assigned = new_cluster_id
            self.cluster_id_updated = True

    def __str__(self):
        if(len(self.point) == 1):
            return '%10.4f --> cluster %d' % (self.point[0], self.cluster_assigned)
        else:
            return '(%10.4f, %10.4f) --> cluster %d' % (self.point[0], self.point[1], self.cluster_assigned)
        
class ClusterMean:
    def __init__(self, dimension):
        self.mean = [0.0] * dimension
        self.data_point_count = 0

    def addDataPoint(self, data_point):
        for index, data_point_dimension in enumerate(data_point):
            self.mean[index] += data_point_dimension

        self.data_point_count += 1

    def setMean(self):
        for index in range(len(self.mean)):
            self.mean[index] /= self.data_point_count

    def getDistance(self, data_point):
        sum = 0.0
        for index, dimension in enumerate(self.mean):
            sum += math.pow(data_point[index] - dimension, 2)
        return math.sqrt(sum)

    def reset(self):
        for index in range(len(self.mean)):
            self.mean[index] = 0.0

        self.data_point_count = 0

    def __str__(self):
        return f"{self.mean}, {self.data_point_count}"

def init_clusters(data_file, K, initialization):
    file = open(data_file, 'r')
    file_data = []

    for fileline in file.readlines():
        fileline = list(map(float, fileline.strip().split()))
        file_data.append(fileline)

    data_points = []
    if(initialization == "round_robin"):
        index = 1
        for data_point in file_data:
            if(index <= K):
                data_points.append(DataPoint(data_point, index))
            else:
                index = 1
                data_points.append(DataPoint(data_point, index))
            
            index += 1
    else:
        for data_point in file_data:
            data_points.append(data_point, random.randint(1, K + 1))

    return data_points

def k_means(data_file, K, initialization):
    data_points = init_clusters(data_file, K, initialization)
    dimensions = len(data_points[0].point)

    cluster_means = []
    for i in range(K):
        cluster_means.append(ClusterMean(dimensions))

    no_changes_in_assignments = False
    while(no_changes_in_assignments == False):
        #Get the means for each cluster
        for cluster_id in range(len(cluster_means)):
            for data_point in data_points:
                if(data_point.cluster_assigned == cluster_id + 1):
                    cluster_means[cluster_id].addDataPoint(data_point.point)

        for cluser_mean in cluster_means:
            cluser_mean.setMean()

        #Reassign
        for data_point in data_points:
            min_dist = math.inf
            min_dist_cluster_id = -1
        
            for cluster_index, mean in enumerate(cluster_means):
                distance = mean.getDistance(data_point.point)

                if(distance < min_dist):
                    min_dist = distance
                    min_dist_cluster_id = cluster_index
            
            data_point.updateAssignedCluster(min_dist_cluster_id + 1)

        #If no updates then break out of the while
        no_changes_in_assignments = True
        for data_point in data_points:
            if(data_point.cluster_id_updated == True):
                no_changes_in_assignments = False
                break

        for cluser_mean in cluster_means:
            cluser_mean.reset()
                            
    for data_point in data_points:
            print(data_point)
