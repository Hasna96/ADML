import numpy as np
from copy import deepcopy
from Attacks.Attack import Attack


class GoodWordAttack(Attack):

    def __init__(self , classifier = None, num_good_words = 500, attack_type = None, name=None):

        super(GoodWordAttack, self).__init__()
        self.num_good_words = num_good_words
        self.features_num = 0
        self.spam_email = None
        self.ham_email = None
        self.spam_class = +1
        self.ham_class = -1
        self.classifier = classifier
        self.attack_type = attack_type
        self.name = name

    def attack(self, test_set, test_labels, malicious_set):
        self.get_spam_ham_emails(test_set, test_labels)
        if self.attack_type == "First N Words":
            good_words = self.first_n_words(self.spam_email, self.ham_email)
            #print(good_words)
        else:
            good_words = self.best_n_words(self.spam_email, self.ham_email)
        adversarial_set = []
        count = 0
        for test_vector in test_set:

            if test_labels[count] == 1:
                adversarial_set.append(self.add_good_words(test_vector, good_words))
                count+=1
            else:
                adversarial_set.append(test_vector.toarray()[0])
                count += 1

        return adversarial_set

    def add_good_words(self, test_vector, good_words):

        feature_vector = deepcopy(test_vector)
        for index in good_words:
            if index not in feature_vector.indices:
                feature_vector.indices = np.append(feature_vector.indices, index)
                feature_vector.data = np.append(feature_vector.data, 1)
                feature_vector.indptr[1] += 1

        return feature_vector.toarray()[0]

    def get_spam_ham_emails(self, test_set, test_labels):

        self.features_num = len(test_set.toarray()[0])
        cond = True
        while cond == True:
            count = 0
            for feature_vector in test_set:
                if test_labels[count]== self.spam_class:
                    self.spam_email = feature_vector
                    cond = False
                    break
                else:
                    count+=1

        cond = True
        while cond == True:
            count = 0
            for feature_vector in test_set:
                if test_labels[count] == self.ham_class:
                    self.ham_email = feature_vector
                    cond = False
                    break
                else:
                    count += 1


    def first_n_words(self, spam_email, ham_email):
        good_words = set()
        spam_email_witness, ham_email_witness = self.find_witness(spam_email, ham_email)

        for index in range(0,self.features_num):
            if  int(spam_email_witness.toarray()[0][index]) == 0:
                self.flip_feature(spam_email_witness, index)
                if self.classifier.predict(spam_email_witness) == self.ham_class:
                    good_words.add(index)
                if len(good_words) == self.num_good_words:
                    return good_words
                self.flip_feature(spam_email_witness, index)

        return good_words


    def best_n_words(self, spam_email, ham_email):
        spam_email_witness, ham_email_witness = self.find_witness(spam_email, ham_email)
        spam_words, ham_words = self.get_word_sets(spam_email_witness, ham_email_witness)
        best_good_words = set()
        iter_wo_change = 0
        max_iter_wo_change = 10
        for word in spam_words:
            flipped = False
            if spam_email_witness.toarray()[0][word] == 1:
                self.flip_feature(spam_email_witness, word)
                flipped = True
            small_words_indices, large_words_indices = self.get_small_large_words(spam_email_witness, ham_words)

            #to keep spam_email_witness the same as the original
            if flipped == True:
                self.flip_feature(spam_email_witness, word)


            if len(best_good_words) + len(large_words_indices) < self.num_good_words:
                ham_words = ham_words - large_words_indices
                best_good_words = best_good_words.union(large_words_indices)
                if len(large_words_indices) == 0:
                    iter_wo_change += 1
                else:
                    iter_wo_change = 0
            else:
                ham_words = ham_words - small_words_indices
                if len(small_words_indices) == 0:
                    iter_wo_change += 1
                else:
                    iter_wo_change = 0

            if iter_wo_change == max_iter_wo_change:
                for i in range(min(self.num_good_words - len(best_good_words), len(ham_words))):
                    best_good_words.add(ham_words.pop())

        return best_good_words


    def get_small_large_words(self, spam_email_witness, ham_words):
        spam_email = deepcopy(spam_email_witness)
        ham_words_indices = ham_words
        small_words = set()
        large_words = set()
        for index in ham_words_indices:
            if int(spam_email.toarray()[0][index]) == 0:
                self.flip_feature(spam_email, index)
                if self.classifier.predict(spam_email) == self.spam_class:
                    small_words.add(index)
                else:
                    large_words.add(index)
                self.flip_feature(spam_email, index)

        #for index in ham_words_indices:
         #   if int(spam_email.toarray()[0][index]) == 0:
          #      self.flip_feature(spam_email, index)
           #     if self.classifier.predict(spam_email) == self.ham_class:
            #        large_words.add(index)
             #   self.flip_feature(spam_email, index)

        return small_words, large_words



    def get_word_sets(self, spam_email_witness, ham_email_witness):
        spam_email = deepcopy(spam_email_witness)
        ham_email = deepcopy(ham_email_witness)
        spam_words = set()
        ham_words = set()
        for index in range (0, self.features_num):

            if int(ham_email.toarray()[0][index]) == 0:
                self.flip_feature(ham_email, index)
                if self.classifier.predict(ham_email) == self.spam_class:
                    spam_words.add(index)
                self.flip_feature(ham_email, index)

        for index in range (0, self.features_num):
            if int(spam_email.toarray()[0][index]) == 0:
                self.flip_feature(spam_email, index)
                if self.classifier.predict(spam_email) == self.ham_class:
                    ham_words.add(index)
                self.flip_feature(spam_email, index)

        return spam_words, ham_words


    def find_witness(self, spam_email, ham_email):
        current_message = deepcopy(ham_email)
        spam_message = spam_email
        spam_words, curr_words = self.get_spam_ham_words(spam_email, ham_email)
        prev_message = None

        while(self.classifier.predict(current_message) != self.spam_class):

            prev_message = deepcopy(current_message)
            word_removed = False

            for index in current_message.indices:
                if index not in spam_words:
                    self.flip_feature(current_message, index)
                    word_removed = True

                    break
            if word_removed: continue

            word_added = False
            for index in spam_message.indices:
                if index not in curr_words:
                    self.flip_feature(current_message, index)
                    curr_words.add(index)
                    word_added = True
                    break

            if not word_added:
                raise Exception('No witness has been found')

        return current_message ,prev_message


    def flip_feature(self, email, feature_index):

        if feature_index in email.indices:
            i = np.argwhere(email.indices == feature_index)
            email.indices = np.delete(email.indices, i[0])
            email.data = np.delete(email.data, 1)
            email.indptr[1] -= 1
        else:
            email.indices = np.append(email.indices, feature_index)
            email.data = np.append(email.data, 1)
            email.indptr[1] += 1



    def get_spam_ham_words(self, spam_email, ham_email):

        spam_email_words = set()
        ham_email_words = set()
        for index in range(0,self.features_num):
            if int(spam_email.toarray()[0][index]) != 0:
                spam_email_words.add(index)

        for index in range(0,self.features_num):
            if int(ham_email.toarray()[0][index]) != 0:
                ham_email_words.add(index)


        return spam_email_words, ham_email_words

    def return_attack(self):
        attack_name = "Good Word"
        return attack_name