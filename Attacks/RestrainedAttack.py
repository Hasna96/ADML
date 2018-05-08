import random
import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix
from Attacks.Attack import Attack

class RestrainedAttack(Attack):

    def __init__(self, classifier = None, attack_power = 0.7, data_movement_factor =0):

        super(RestrainedAttack, self).__init__()
        self.features_num =0
        self.attack_power = attack_power
        self.classifier = classifier
        self.data_movement_factor = data_movement_factor
        self.innocuous_target = None

    def attack(self, test_set, test_labels, malicious_set):

        self.features_num = len(malicious_set[0])

        adversarial_set = []
        for mal in malicious_set:
            adversarial_set.append(self.transform_feature_vector(mal))
        return adversarial_set



    def set_innocuous_target(self, train_set, train_labels, learner, type):
        if type == 'random':
            cond = True
            while cond == True:
                count = 0
                for feature_vector in train_set:
                    if train_labels[count] == -1:
                        self.innocuous_target = feature_vector
                        cond = False
                        break
                    else:
                        count += 1
        elif type == 'centroid':
            target = self.find_centroid(train_set, train_labels)
            if learner.predict(target) == 1:
                #print("Fail to find centroid from estimated training data")
                cond = True
                while cond == True:
                    count = 0
                    for feature_vector in train_set:
                        if train_labels[count] == -1:
                            self.innocuous_target = feature_vector
                            cond = False
                            break
                        else:
                            count += 1
            else:
                self.innocuous_target = target

    def find_centroid(self, train_set, train_labels):
        self.features_num = len(train_set[0].toarray()[0])
        indices = []
        data = []
        for index in range(0, self.features_num):
            sum = 0
            count = 0
            for feature_vector in train_set:
                if train_labels[count] == -1:
                    sum += feature_vector.toarray()[0][index]
                    count +=1
                else:
                    count +=1
            sum /= self.features_num
            if sum != 0:
                indices.append(index)
                data.append(sum)
        indptr = [0, len(indices)]
        centroid = csr_matrix((data, indices, indptr), shape=(1, self.features_num))
        return centroid

    def transform_feature_vector(self, feature_vector):

        feature_vector_copy = deepcopy(feature_vector)
        for index in range(0, self.features_num):
            #if index in transform_features_index:
                xij =  feature_vector_copy[index]
                target = self.innocuous_target.toarray()[0][index]
                if abs(xij) + abs(target) == 0:
                    bound = 0
                else:
                    bound = self.data_movement_factor * (1 - self.attack_power *(abs(target - xij)
                            /(abs(xij) + abs(target)))) * abs((target - xij))
                delta_ij = random.uniform(0, bound)
                feature_vector_copy[index] = xij + delta_ij

        return feature_vector_copy

    def get_malicious_set(self, real_labels, predicted_labels, test_set):

        count = 0
        malicious_set = []
        mal_labels = []

        for feature_vector in test_set.toarray():

            if real_labels[count] == 1:
                malicious_set.append(feature_vector)
                mal_labels.append(real_labels[count])
            count += 1


        return malicious_set, mal_labels

    def return_attack(self):
        attack_name = "Restrained"
        return attack_name



