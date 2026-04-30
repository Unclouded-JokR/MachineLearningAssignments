from decision_tree import *
import cProfile

# When you test your code, you can change this line to reflect where the 
# dataset directory is located on your machine.
dataset_directory = "uci_data"

# When you test your code, you can select the dataset you want to use 
# by modifying the next lines
#dataset = "pendigits_string"
#dataset = "satellite"
dataset = "yeast"


training_file = dataset_directory + "/" + dataset + "_training.txt"
test_file = dataset_directory + "/" + dataset + "_test.txt"

# When you test your code, you can select the function arguments you want to use 
# by modifying the next lines
#option = "optimized"
option = 20
#option = 3
#option = 15
pruning_thr = 5

# test_accuracies = []
# for option in [1, 3, 5, 10, 15, 20]:
#     for pruning_threshold in [1, 5, 10, 20, 30, 50, 100]:
#         accuracies = []
#         for i in range(10):
#             accuracy = decision_tree(training_file, test_file, option, pruning_thr)
#             accuracies.append(accuracy)
#         mean_acc = np.mean(accuracies)
#         std_dev = np.std(accuracies)
#         print(f"Threshold: {pruning_threshold} | Accuracies: {accuracies}")
#         print(f"Average: {mean_acc:.4f} | Std Dev: {std_dev:.4f}\n")
#         test_accuracies.append((sum(accuracies)/10, std_dev))

# print(test_accuracies)

decision_tree(training_file, test_file, option, pruning_thr)
# def main():
#     decision_tree(training_file, test_file, option, pruning_thr)


# if __name__ == '__main__':
#     cProfile.run('main()')