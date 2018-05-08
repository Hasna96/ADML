import os
import email
import codecs
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

class Emails(object):

    def __init__(self, emails_path = None, max_features =3000):

        super(Emails, self).__init__()

        self.emails_path = emails_path
        self.max_features = max_features
        self.raw_emails_bodies, self.file_names= self.get_raw_email(emails_path)
        self.feature_vector =self.get_feature_vector(self.raw_emails_bodies)

    def get_raw_email(self, emails_path):
        #list to store raw emails
        raw_emails_bodies = []
        filenames = []
        for file in tqdm(os.listdir(emails_path), desc=" Extracting raw email bodies"):
            filenames.append(file)
            email_path = emails_path +"/" +file
            with codecs.open(email_path, 'r', encoding="utf8", errors='ignore') as fp:
                email_msg = fp.read()
                email_string = email.message_from_string(email_msg)
                if email_string.is_multipart():
                    initial_body = ""
                    for payload in email_string.get_payload():

                        body = payload.get_payload()
                        body = (re.sub("<.*?>", "", str(body))).replace('\n\t', ' ').replace('\n', ' ')
                        initial_body += body + ""
                    raw_emails_bodies.append(initial_body)
                else:

                    body = email_string.get_payload()
                    body = (re.sub("<.*?>", "", str(body))).replace('\n\t', ' ').replace('\n', ' ')
                    raw_emails_bodies.append(body)

        return raw_emails_bodies, filenames


    def get_feature_vector(self, emails_bodies):

        vectorizer = TfidfVectorizer(analyzer='word', strip_accents=None,
                                     ngram_range=(1, 1), max_df=1.0,
                                     min_df=1, max_features=self.max_features,
                                     binary=False, stop_words='english',
                                     use_idf=True, norm=None)

        vectorizer = vectorizer.fit(emails_bodies)

        features = vectorizer.transform(tqdm(emails_bodies, desc=" Creating emails feature vector"))

        return features