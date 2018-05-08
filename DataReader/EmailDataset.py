import os
import email
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class EmailDataset(object):

    def __init__(self, emails_path = None, labels_path=None, max_features =3000, test_size=0.3, raw=True, input_file=None):

        super(EmailDataset, self).__init__()
        self.raw=raw
        self.input_file = input_file
        if self.raw == True :
            self.emails_path = emails_path
            self.labels_path = labels_path
            self.max_features = max_features
            self.test_size = test_size
            self.labels = self.get_labels(self.labels_path)
            self.raw_emails_bodies = self.get_emails_body(self.emails_path)
            self.features, self.feature_names, self.features_with_indices, self.features_weights , self.features_bin = \
                self.get_feature_vectors(self.raw_emails_bodies)
            self.x_train, self.x_test, self.y_train, self.y_test =self.split_data(self.features,self.labels, self.test_size)
            self.x_train_bin, self.x_test_bin, self.y_train, self.y_test =self.split_data(self.features_bin,self.labels, self.test_size)
        else:
            self.emails_path, self.labels_path, self.max_features, self.test_size, self.labels, self.raw_emails_bodies,\
            self.features, self.feature_names, self.features_with_indices, self.features_weights, self.features_bin, self.x_train,\
            self.x_test, self.y_train, self.y_test, self.x_train_bin, self.x_test_bin = self.load_saved_dataset(self.input_file)


    #function to load all the emails in a specific directory and convert them to strings
    #returns a list of all emails as strings
    def load_emails(self, emails_path):
        #list to store the emails
        emails = []
        #loop through each file in the directory
        for file in os.listdir(emails_path):
            email_path = emails_path +"/" +file
            with open(email_path, 'r', encoding="utf8", errors='ignore') as fp:
                email_msg = fp.read()
                emails.append(email_msg)

        return emails

    #function to extract the body part from the emails
    #return a list of emails body part
    def get_emails_body(self, emails_path):
        #list to store the emails body
        emails_body = []
        #load the raw emails
        emails = self.load_emails(emails_path)
        for file in tqdm(emails, desc=" Extracting raw email bodies"):
            #convert the email string into a message object structure of class EmailMessage
            email_msg = email.message_from_string(file)
            if email_msg.is_multipart():
                initial_body = ""
                #loop through the payload since the message has multiple payloads
                for payload in email_msg.get_payload():
                    body = payload.get_payload()
                    #regex to remove HTML tags from the message body
                    body = (re.sub("<.*?>", "", str(body))).replace('\n\t', ' ').replace('\n', ' ')
                    initial_body += body + ""
                emails_body.append(initial_body)
            else:
                body = email_msg.get_payload()
                body = (re.sub("<.*?>", "", str(body))).replace('\n\t', ' ').replace('\n', ' ')
                emails_body.append(body)

        return emails_body

    #function to load the email labels
    #returns a list of labels, where the index of each element specifics the file
    def get_labels(self, labels_path):
        #list to store emails labels
        labels = []
        with open(labels_path, 'r') as email_labels:
            for line in tqdm(email_labels, desc="\n Creating emails label list"):
                #change
                if int(line[0]) == 0:
                    labels.append(+1)
                else:
                    labels.append(-1)

        return labels
    #function to transform the raw email bodies into feature vectors
    def get_feature_vectors(self, emails_bodies):
        #create a vectoriser
        vectorizer = TfidfVectorizer(analyzer='word', strip_accents=None,
                                     ngram_range=(1, 1), max_features=self.max_features,
                                     stop_words='english',norm=None)
        #train it on the emails body
        vectorizer = vectorizer.fit(emails_bodies)
        #transform the raw emails body into feature vectors
        features_vectors = vectorizer.transform(tqdm(emails_bodies, desc=" Creating emails feature vector"))
        #created a binarizer that turns the TF-IDF features into binary feature vectors
        # (0 for non occurance and 1 for occurance)
        binarizer = Binarizer().fit(features_vectors)
        #needed for good word attack
        features_bin = binarizer.transform(features_vectors)

        #get the feature names, vocabulary and weights
        feature_names = vectorizer.get_feature_names()
        features_with_indices = vectorizer.vocabulary_
        features_weights = vectorizer.idf_

        return features_vectors, feature_names, features_with_indices, features_weights, features_bin

    #function to split the features and labels into training and testing subsets
    def split_data(self, features, labels, test_size =0.3):

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

        return  x_train, x_test, y_train, y_test


    def save_dataset(self, emaildataset, output_path):

        with open(output_path + ".pkl", 'wb') as output:
            pickle.dump(emaildataset, output, pickle.HIGHEST_PROTOCOL)


    def load_saved_dataset(self, input_path):

        with open(input_path, 'rb') as input:
            dataset = pickle.load(input)


        return  dataset.emails_path, dataset.labels_path, dataset.max_features,dataset.test_size, dataset.labels, dataset.raw_emails_bodies, \
                dataset.features, dataset.feature_names, dataset.features_with_indices, dataset.features_weights, dataset.features_bin, dataset.x_train, \
                dataset.x_test, dataset.y_train,dataset.y_test, dataset.x_train_bin, dataset.x_test_bin






