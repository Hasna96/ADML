import numpy as np
from copy import deepcopy
from Attacks.Attack import Attack

class FeatureDeletionAttack(Attack):

    def __init__(self , classifier, del_num, name):

        super(FeatureDeletionAttack, self).__init__()
        self.classifier = classifier
        self.del_num = del_num
        self.features_num = 0
        self.name = name



    def attack(self, test_set, test_labels, malicious_set):
        del_features_index = self.get_del_features_index(self.classifier, self.del_num)
        self.features_num = len(malicious_set[0])
        adversarial_set = []

        for mal in malicious_set:

            adversarial_set.append(self.del_features(mal,del_features_index))

        return adversarial_set

    def get_del_features_index(self, trained_classifier, del_num):

        if self.name == "MultinomialNB":
            del_features_index = np.flipud(np.argsort(trained_classifier.coef_[0]))[:del_num]
            return del_features_index
        elif self.name == "SVM":
            del_features_index = np.flipud(np.argsort(trained_classifier.coef_[0].toarray()[0]))[:del_num]
            return del_features_index
        elif self.name == "DT":
            del_features_index = np.flipud(np.argsort(trained_classifier.feature_importances_))[:del_num]

            return del_features_index



    def del_features(self, feature_vector, del_features_index):

        feature_vector_copy = deepcopy(feature_vector)
        for index in range(0, self.features_num):

            if index in del_features_index:
                feature_vector_copy[index] = 0

        return  feature_vector_copy


    def get_malicious_set(self, real_labels, predicted_labels, test_set):

        count = 0
        malicious_set = []
        mal_labels = []

        for feature_vector in test_set.toarray():

            #if predicted_labels[count] == 1 and real_labels[count] == 1:
            if real_labels[count] == 1:
                malicious_set.append(feature_vector)
                mal_labels.append(real_labels[count])
            count += 1
        #added
        self.labels = mal_labels
        return malicious_set, mal_labels

    def return_attack(self):
        attack_name = "Feature Deletion"
        return attack_name







