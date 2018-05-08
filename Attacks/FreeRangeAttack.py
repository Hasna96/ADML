import random
import numpy as np
from copy import deepcopy
from Attacks.Attack import Attack

class FreeRangeAttack(Attack):

    def __init__(self, classifier = None, attack_power = 1):

        super(FreeRangeAttack, self).__init__()
        self.features_num =0
        self.attack_power = attack_power
        self.classifier = classifier


    def attack(self, test_set, test_labels, malicious_set):

        self.features_num = len(malicious_set[0])
        max_values = self.max_feature_val(test_set)
        min_values = self.min_feature_val(test_set)
        adversarial_set = []
        for mal in malicious_set:
            adversarial_set.append(self.transform_feature_vector(mal, min_values, max_values))
        return adversarial_set



    #find the maximum value that a feature can have in all the feature vectors
    def max_feature_val(self, test_set):

        feature_indices = []
        feature_max_values = []
        for index in range(self.features_num):
            max = 0
            for feature_vector in test_set.toarray():
                feature_value = feature_vector[index]
                if int(feature_value) >= max:
                    max = feature_value

            feature_indices.append(index)
            feature_max_values.append(max)

        return feature_max_values

    def min_feature_val(self, test_set):

        feature_indices = []
        feature_min_values = []
        for index in range(self.features_num):
            min = 1000
            for feature_vector in test_set.toarray():
                feature_value = feature_vector[index]
                if int(feature_value) <= min:
                    min = feature_value

            feature_indices.append(index)
            feature_min_values.append(min)

        return feature_min_values

    #function to transform feature vectors into adversarial examples
    def transform_feature_vector(self, feature_vector, min_values, max_values):

        feature_vector_copy = deepcopy(feature_vector)
        for index in range(0, self.features_num):
            #get the jth feature value
            xij =  feature_vector_copy[index]
            #set the lower and upper bounds of the jth feature value
            lower_bound = self.attack_power * (min_values[index] - xij)
            upper_bound = self.attack_power * (max_values[index] - xij)
            #find delta_ij
            delta_ij = random.uniform(lower_bound, upper_bound)
            #add delta_ij value to the original value of the jth feature
            feature_vector_copy[index] = xij + delta_ij


        return feature_vector_copy


    def get_malicious_set(self, real_labels, predicted_labels, test_set):

        count = 0
        malicious_set = []
        mal_labels = []

        for feature_vector in test_set.toarray():
            #predicted_labels[count] == 1 and
            if real_labels[count] == 1:
                malicious_set.append(feature_vector)
                mal_labels.append(real_labels[count])
            count += 1


        return malicious_set, mal_labels

    def return_attack(self):
        attack_name = "Free Range"
        return attack_name