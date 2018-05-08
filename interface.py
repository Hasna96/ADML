from colorama import Fore
from colorama import init
from Classifiers.SupporVectorMachinesClassifier import svmClassifier
from Classifiers.DecisionTreeClassifier import decisionTreeClassifier
from Classifiers.NaiveBayesClassifier import multiNomialNaiveBayesClassifier
from DataReader.EmailDataset import EmailDataset
from DataReader.Email import Emails
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, accuracy_score
from Attacks.FeatureDeletionAttack import FeatureDeletionAttack
from Attacks.FreeRangeAttack import FreeRangeAttack
from Attacks.GoodWordAttack import GoodWordAttack
from Attacks.RestrainedAttack import RestrainedAttack
from sklearn import tree

import os
import time
import random
import pickle
import warnings


#remove sklearn precision warning
warnings.filterwarnings("ignore")


def banner():
    os.system('cls')
    init()
    print(Fore.CYAN + "###################################################################################################################")
    print(Fore.CYAN + '#' + Fore.LIGHTCYAN_EX + '        Adversarial Reasoning in Machine Learning for Natural Language Processing: The Case of Spam Emails       ' + Fore.CYAN + '#')
    print(Fore.CYAN + "###################################################################################################################")
    print(Fore.CYAN + "-------------------------------------------------------------------------------------------------------------------")


def instructions():
    print("\n")
    print(Fore.LIGHTGREEN_EX + " Welcome to Adversarial Reasoning in Machine Learning for Natural "
                               "Language Processing: The Case of Spam Emails!")
    print(Fore.LIGHTRED_EX + "\n *Important*" + Fore.RESET + " This software requires specifying the path"
          + " of the folder which holds the Spam and Ham emails and the" + "\n path of the file which holds "
            "the labels of those emails for creating preprocessed dataset for training and \n testing "
            "classifiers. If you already have a saved dataset, please select no when asked for creating "
             "new dataset\n and specify the path to the dataset.")

def set_dataset():
    print(Fore.RESET + "\n - Would you like to create a new dataset from raw emails? (" + Fore.LIGHTGREEN_EX + "Y" + Fore.RESET + "/" + Fore.LIGHTRED_EX + "N" + Fore.RESET + ")")
    decision = input("   ")
    if decision == "Y" or decision == "y":
        emails_path, labels_path = get_files_path(decision)
        print(Fore.GREEN + "\n -------------------------------- " + Fore.LIGHTGREEN_EX + "Please wait while a new email dataset is being created" + Fore.GREEN + "-----------------------------")
        e = EmailDataset(emails_path, labels_path)
        save_dataset(e)
        return e

    elif decision == "N" or decision == "n":
        dataset_path = get_files_path(decision)
        print(Fore.GREEN + "\n -------------------------------- " + Fore.LIGHTGREEN_EX + "Please wait while your dataset is getting loaded" + Fore.GREEN + "-----------------------------")
        e = EmailDataset(input_file=dataset_path, raw=False)
        time.sleep(5)
        print(Fore.RESET + "\n - Successfully loaded the dataset.")
        return e

    else:
        print(Fore.LIGHTRED_EX + " Error: "+Fore.RESET + " You can only choose Y or N, please try again")
        set_dataset()

def get_files_path(decision):
    if decision == "Y" or decision == "y":
        emails_path = input(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Please specify the path of the folder that holds the raw emails: ")
        while not os.path.isdir(emails_path):
            emails_path = input(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " The path you have specified does not exist, please try again: ")
        labels_path = input( Fore.LIGHTCYAN_EX + " (2)" + Fore.RESET + " Please specify the path of the folder that holds the emails labels: ")
        while not os.path.isfile(labels_path):
            labels_path = input(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " The path you have specified does not exist, please try again: ")

        return emails_path, labels_path
    elif decision == "N" or decision == "n":
        dataset_path = input(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Please specify the path of the existing dataset file (.pkl): ")
        dataset_path = "DataReader/Data/" + dataset_path
        while not os.path.isfile(dataset_path):
            dataset_path = input(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " The path you have specified does not exist, please try again: ")

        return dataset_path

    else:
        print(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " You can only choose Y or N, please try again")
        set_dataset()

def save_dataset(dataset):
    time.sleep(5)
    print(Fore.RESET + "\n - Successfully created a new email dataset. Would you like to save it? (" + Fore.LIGHTGREEN_EX + "Y" + Fore.RESET + "/" + Fore.LIGHTRED_EX + "N" + Fore.RESET + ")")
    decision = input("   ")
    if decision == "Y" or decision == "y":
        output_path = input( Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Please specify where you would like to save the file: ")
        output_path = "DataReader/Data/" + output_path
        dataset.save_dataset(dataset, output_path)
        while not os.path.isfile(output_path + ".pkl"):
            output_path = input(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " Could not save the dataset, please try again: ")
            output_path = "DataReader/Data/" + output_path
            dataset.save_dataset(dataset, output_path)
        time.sleep(5)
        print(" - Successfully saved the dataset in the desired path")

    elif decision == "N" or decision == "n":
        pass

    else:
        print(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " You can only choose Y or N, please try again")
        save_dataset(dataset)

def save_ad_set(arr):
    time.sleep(2)
    decision = input("   ")
    if decision == "Y" or decision == "y":
        output_path = input(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Please specify where you would like to save the file: ")
        output_path = "DataReader/Data/" +output_path
        with open(output_path + ".pkl", 'wb') as output:
            pickle.dump(arr, output, pickle.HIGHEST_PROTOCOL)
        while not os.path.isfile(output_path + ".pkl"):
            output_path = input(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " Could not save the dataset, please try again: ")
            output_path = "DataReader/Data/" + output_path
            save_ad_set(arr)
        time.sleep(2)
        print(" - Successfully saved the adversarial set in the desired path")

    elif decision == "N" or decision == "n":
        pass

    else:
        print(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " You can only choose Y or N, please try again")
        save_ad_set(arr)


def save_model(classifier_norm, classifier_bin, name):
    classifier = [classifier_norm, classifier_bin, name]
    time.sleep(2)

    output_path = input(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Please specify where you would like to save the classifier: ")
    output_path = "DataReader/Data/" + output_path
    with open(output_path + ".pkl", 'wb') as output:
        pickle.dump(classifier, output, pickle.HIGHEST_PROTOCOL)
    while not os.path.isfile(output_path + ".pkl"):
        output_path = input(
            Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " Could not save the model, please try again: ")
        save_model(classifier_norm, classifier_bin, name)
    time.sleep(6)
    print(" - Successfully saved the model in the desired path")



def load_model():

    input_path = input(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Please specify the name of the existing classifier file (.pkl)")
    input_path = "DataReader/Data/" + input_path
    while not os.path.isfile(input_path):
        input_path = input(
            Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " The path you have specified does not exist, please try again: ")
        input_path = "DataReader/Data/" + input_path
    with open(input_path, 'rb') as input_file:
        model = pickle.load(input_file)

    return model




def show_dataset(dataset):
    print(Fore.CYAN + "-------------------------------------------------------------------------------------------------------------------")
    print(Fore.LIGHTMAGENTA_EX + "                        Dataset: " + Fore.RESET + str(
        (dataset.x_train).shape[0]) + " Training emails - " + str((dataset.x_test).shape[0]) + " Testing emails - " + str(
        (dataset.x_test).shape[0] + (dataset.x_train).shape[0]) + " emails in total")
    print( Fore.LIGHTCYAN_EX + "-------------------------------------------------------------------------------------------------------------------")

def show_banner2(title):
    print(Fore.LIGHTMAGENTA_EX + "                                                "+str(title)+"")


def options(screen):
    if screen == 2:
        print(Fore.LIGHTGREEN_EX + " Choose from the following available options:")
        print(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Create a new Classifier " + Fore.CYAN + "-- will create a new Classifier and train it ( Naive Bayes| Decision Tree| SVM)")
        print(Fore.LIGHTCYAN_EX + " (2)" + Fore.RESET + " Create and test all classifiers ")
        print(Fore.LIGHTCYAN_EX + " (3)" + Fore.RESET + " Create and attack all classifier with all possible attacks ")
        print(Fore.LIGHTCYAN_EX + " (4)" + Fore.RESET + " Change Dataset")
    elif screen == 4:
        print(Fore.LIGHTGREEN_EX + " Choose from the following available options:")
        print(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Naive Bayes Classifier ")
        print(Fore.LIGHTCYAN_EX + " (2)" + Fore.RESET + " Decision Tree Classifer")
        print(Fore.LIGHTCYAN_EX + " (3)" + Fore.RESET + " Support Vector Machines Classifier")
        print(Fore.LIGHTCYAN_EX + " (4)" + Fore.RESET + " Black Box Classifier")
        print(Fore.LIGHTCYAN_EX + " (5)" + Fore.RESET + " Load Existing Classifier")
        print(Fore.LIGHTCYAN_EX + " (6)" + Fore.RESET + " Go back to main menu")
    elif screen == 5:
        print(Fore.LIGHTGREEN_EX + " Choose from the following available options:")
        print(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Test Classifier ")
        print(Fore.LIGHTCYAN_EX + " (2)" + Fore.RESET + " Predict the label of Unseen Email")
        print(Fore.LIGHTCYAN_EX + " (3)" + Fore.RESET + " Attack Classifier")
        print(Fore.LIGHTCYAN_EX + " (4)" + Fore.RESET + " Save Classifier")
        print(Fore.LIGHTCYAN_EX + " (5)" + Fore.RESET + " Go back to classifiers menu")
    elif screen == 6:
        print(Fore.LIGHTGREEN_EX + " Choose from the following available options:")
        print(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Good word Attack " )
        print(Fore.LIGHTCYAN_EX + " (2)" + Fore.RESET + " Free range Attack")
        print(Fore.LIGHTCYAN_EX + " (3)" + Fore.RESET + " Restrained Attack")
        print(Fore.LIGHTCYAN_EX + " (4)" + Fore.RESET + " Feature deletion Attack")
        print(Fore.LIGHTCYAN_EX + " (5)" + Fore.RESET + " Load saved Attack")
        print(Fore.LIGHTCYAN_EX + " (6)" + Fore.RESET + " Go back to classifier page")
    elif screen == 9:
        print(Fore.LIGHTGREEN_EX + " Choose from the following available options:")
        print(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Attack Classifier")
        print(Fore.LIGHTCYAN_EX + " (2)" + Fore.RESET + " Go back to main menu")

def screen1():
    banner()
    instructions()
    e = set_dataset()
    return e
def screen2(dataset):
    banner()
    show_banner2("Main Menu")
    show_dataset(dataset)
    options(2)
    choice = input(" Your choice: ")
    if choice == "1":
        screen3(dataset)
    elif choice == "2":
        summary_screen(dataset)
    elif choice =="3":
        attack_summary_screen(dataset)
    elif choice == "4":
        init_app()
    else:
        pass

def screen3(dataset):
    banner()
    show_banner2("Classifiers")
    show_dataset(dataset)
    options(4)
    choice = input(" Your choice: ")
    if choice == "1":
        clf = multiNomialNaiveBayesClassifier()
        clf_normal = clf.fit(dataset.x_train, dataset.y_train)
        clf_bin = clf.fit(dataset.x_train_bin, dataset.y_train)
        name = clf.get_alg()
        screen4(dataset, clf_normal, clf_bin, name)
    elif choice == "2":
        clf = decisionTreeClassifier()
        clf_normal = clf.fit(dataset.x_train, dataset.y_train)
        clf_bin = clf.fit(dataset.x_train_bin, dataset.y_train)
        name = clf.get_alg()
        #tree.export_graphviz(clf_normal, class_names=["ham", "spam"],  out_file = 'tree.dot')
        #tree.export_graphviz(clf_bin, class_names = ["ham", "spam"],out_file='treebin.dot')
        screen4(dataset, clf_normal, clf_bin, name)
    elif choice == "3":
        clf = svmClassifier()
        clf_normal = clf.fit(dataset.x_train, dataset.y_train)
        clf_bin = clf.fit(dataset.x_train_bin, dataset.y_train)
        name = clf.get_alg()
        screen4(dataset, clf_normal, clf_bin, name)
    elif choice == "4":
        classifiers = [multiNomialNaiveBayesClassifier(), decisionTreeClassifier(), svmClassifier()]
        clf = random.choice(classifiers)
        clf_normal = clf.fit(dataset.x_train, dataset.y_train)
        clf_bin = clf.fit(dataset.x_train_bin, dataset.y_train)
        name = clf.get_alg()
        screen9(dataset, clf_normal, clf_bin, name)

    elif choice == "5":
        clf=load_model()
        screen4(dataset, clf[0], clf[1], clf[2])

    elif choice == "6":
        screen2(dataset)
    else:
        pass
def screen4(dataset, classifier_norm, classifier_bin, name):
    banner()
    show_banner2("Classifier: "+  name )
    show_dataset(dataset)
    options(5)
    choice = input(" Your choice: ")
    if choice == "1":
        pred = classifier_norm.predict(dataset.x_test)
        pred_bin = classifier_bin.predict(dataset.x_test_bin)
        screen5(dataset, classifier_norm, classifier_bin, name, pred, pred_bin)
    elif choice == "2":
        screen8(dataset, classifier_norm, classifier_bin, name)
    elif choice == "3":
        pred = classifier_norm.predict(dataset.x_test)
        screen6(dataset, classifier_norm, classifier_bin, pred, name)

    elif choice == "4":
        save_model(classifier_norm, classifier_bin, name)
        screen4(dataset, classifier_norm, classifier_bin, name)

    elif choice == "5":
        screen3(dataset)
    else:
        pass
def screen9(dataset, classifer_norm, classifier_bin, name):
    banner()
    show_banner2("Classifier")
    show_dataset(dataset)
    options(9)
    choice = input(" Your choice: ")
    if choice == "1":
        screen10(dataset, classifer_norm, classifier_bin, name="Black")
    elif choice == "2":
        screen2(dataset)
    else:
        pass

def screen10(dataset, classifier_norm, classifier_bin, name):
    banner()
    show_banner2("Attack evaluation report")
    show_dataset(dataset)
    ad_path = input( Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Please specify the path of the file that holds the saved adversarial set: ")
    ad_path = "DataReader/Data/" + ad_path
    while not os.path.isfile(ad_path):
        ad_path = input(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " The path you have specified does not exist, please try again: " + Fore.GREEN + " ")
        ad_path = "DataReader/Data/" + ad_path
    with open(ad_path, 'rb') as input2:
        ad_set = pickle.load(input2)

    if ad_set[2] == "gw" or ad_set[2] == "fd":
        ##print(ad_set[2])
        ad_preds = classifier_bin.predict(ad_set[0])
    else:
        ad_preds = classifier_norm.predict(ad_set[0])

    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(
        confusion_matrix(ad_set[1], ad_preds, labels=[-1, 1])))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(
        accuracy_score(ad_set[1], ad_preds)))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(
        precision_score(ad_set[1], ad_preds, average='binary')))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(
        recall_score(ad_set[1], ad_preds, average='binary')))

    go_bk(dataset, classifier_norm, classifier_bin, name)

def screen8(dataset, classifer_norm, classifier_bin, name):
    banner()
    show_banner2("Predict emails labels")
    show_dataset(dataset)
    emails_path = input(Fore.LIGHTCYAN_EX + " (1)" + Fore.RESET + " Please specify the path of the folder that holds the raw emails: ")
    while not os.path.isdir(emails_path):
        emails_path = input(Fore.LIGHTRED_EX + " Error: " + Fore.RESET + " The path you have specified does not exist, please try again: " + Fore.GREEN +" ")

    emails = Emails(emails_path)
    preds = classifer_norm.predict(emails.feature_vector)
    print(Fore.LIGHTCYAN_EX + " (*)" + Fore.RESET + " Predicted emails labels: ")
    count = 0
    for file in emails.file_names:
        print(Fore.LIGHTCYAN_EX + " " + Fore.RESET + file +": "+ str(preds[count]))
        count+=1

    go_bk(dataset, classifer_norm, classifier_bin, name)

def screen5(dataset, classifier_norm, classifier_bin, name, preds, preds_bin):
    banner()
    show_banner2("Classifier test evaluation report")
    show_dataset(dataset)
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX +str(confusion_matrix(dataset.y_test, preds)))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX +str(accuracy_score(dataset.y_test, preds)))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX +str(precision_score(dataset.y_test, preds, average='binary')))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX +str(recall_score(dataset.y_test, preds, average='binary')))
    #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "The Receiver Operating Characteristic Curve (ROC AUC): " + Fore.LIGHTGREEN_EX +str(roc_auc_score(dataset.y_test, preds)))
    go_bk(dataset, classifier_norm, classifier_bin, name)
    #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX +str(confusion_matrix(dataset.y_test, preds_bin)))
    #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX +str(accuracy_score(dataset.y_test, preds_bin)))
    #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX +str(precision_score(dataset.y_test, preds_bin, average='binary')))
    #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX +str(recall_score(dataset.y_test, preds_bin, average='binary')))
    #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "The Receiver Operating Characteristic Curve (ROC AUC): " + Fore.LIGHTGREEN_EX +str(roc_auc_score(dataset.y_test, preds_bin)))



    #go_bk(dataset)

def screen6(dataset, classifier_norm, classifier_bin, pred, name):
    banner()
    show_banner2("Attacks")
    show_dataset(dataset)
    options(6)
    choice = input(" Your choice: ")
    if choice == "1":

        attack = GoodWordAttack(classifier=classifier_bin, attack_type="First N Words", name=name)
        ad_set = attack.attack(test_set=dataset.x_test_bin, test_labels=dataset.y_test, malicious_set=None)
        ad_preds = classifier_bin.predict(ad_set)
        atk_name ="gw"
        screen7(dataset, ad_set, ad_preds, classifier_norm, classifier_bin, dataset.y_test, atk_name, name)

    elif choice == "2":
        attack = FreeRangeAttack(classifier_norm)
        mal_set, mal_labels = attack.get_malicious_set(dataset.y_test, pred, dataset.x_test)
        ad_set = attack.attack(test_set=dataset.x_test, malicious_set=mal_set, test_labels=None)
        ad_preds =classifier_norm.predict(ad_set)
        atk_name = "fr"
        screen7(dataset, ad_set, ad_preds, classifier_norm, classifier_bin, mal_labels, atk_name, name)

    elif choice == "3":
        attack = RestrainedAttack(classifier_norm)
        mal_set, mal_labels = attack.get_malicious_set(dataset.y_test, pred, dataset.x_test)
        attack.set_innocuous_target(dataset.x_train, dataset.y_train, classifier_norm, "random")
        ad_set = attack.attack(malicious_set=mal_set, test_set=None, test_labels=None)
        ad_preds = classifier_norm.predict(ad_set)
        atk_name = "rr"
        screen7(dataset, ad_set, ad_preds, classifier_norm, classifier_bin, mal_labels, atk_name, name)

    elif choice == "4":
        attack = FeatureDeletionAttack(classifier_bin, 1000, name=name)
        mal_set, mal_labels = attack.get_malicious_set(dataset.y_test, pred, dataset.x_test_bin)
        ad_set = attack.attack(malicious_set=mal_set, test_set=None, test_labels=None)
        ad_preds = classifier_bin.predict(ad_set)
        atk_name = "fd"
        screen7(dataset, ad_set, ad_preds, classifier_norm, classifier_bin, mal_labels, atk_name, name)

    elif choice == "5":
        screen10(dataset, classifier_norm, classifier_bin, name)
    elif choice == "6":
        screen4(dataset, classifier_norm, classifier_bin , name)
    else:
        pass

def screen7(dataset, ad_set, ad_preds, classifier_norm, classifier_bin,  preds, atk_name, name):
    banner()
    show_banner2("Attack evaluation report")
    show_dataset(dataset)
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(confusion_matrix(preds, ad_preds, labels=[-1, 1])))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(accuracy_score(preds, ad_preds)))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(precision_score(preds, ad_preds, average='binary')))
    print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(recall_score(preds, ad_preds, average='binary')))

    print(Fore.RESET + "\n - Successfully created a new adversarial set. Would you like to save it? (" + Fore.LIGHTGREEN_EX + "Y" + Fore.RESET + "/" + Fore.LIGHTRED_EX + "N" + Fore.RESET + ")")
    arr = [None,None,None]
    arr[0] = ad_set
    arr[1] = preds
    arr[2] = atk_name
    save_ad_set(arr)

    go_bk(dataset, classifier_norm, classifier_bin, name)

def go_bk(dataset, classifier_norm, classifier_bin, name):

    print(Fore.RESET + "\n - Would you like to go back to your classifier page? (" + Fore.LIGHTGREEN_EX + "Y" + Fore.RESET + "/" + Fore.LIGHTRED_EX + "N" + Fore.RESET + ")")
    decision = input("   ")
    if decision == "Y" or decision == "y":
        if name == "Black":
            screen9(dataset,  classifier_norm, classifier_bin, name)
        else:
            screen4(dataset,  classifier_norm, classifier_bin, name)

    elif decision == "N" or decision == "n":
        screen2(dataset)
    else:
        #go_bk(dataset, classifier_norm, classifier_bin, name)
        pass

def go_bk_main(dataset):
    print(Fore.RESET + "\n - Would you like to go back to main menu? (" + Fore.LIGHTGREEN_EX + "Y" + Fore.RESET + "/" + Fore.LIGHTRED_EX + "N" + Fore.RESET + ")")
    decision = input("   ")
    if decision == "Y" or decision == "y":
        screen2(dataset)
    elif decision == "N" or decision == "n":
        init_app()
    else:
        #go_bk(dataset, classifier_norm, classifier_bin, name)
        pass

def summary_screen(dataset):
    banner()
    show_banner2("Evaluation summary on all classifiers")
    show_dataset(dataset)

    classifiers = [multiNomialNaiveBayesClassifier(), decisionTreeClassifier(), svmClassifier()]
    names = ["Naive Bayes Classifier:", "Decision Tree Classifier:", "Support Vector Machines Classifier:"]
    count = 0
    for classifier in classifiers:
        clf = classifier
        clf.fit(dataset.x_train, dataset.y_train)
        pred = clf.predict(dataset.x_test)
        print(Fore.LIGHTCYAN_EX + names[count] + "\n")
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(confusion_matrix(dataset.y_test, pred)))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(accuracy_score(dataset.y_test, pred)))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(precision_score(dataset.y_test, pred, average='binary')))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(recall_score(dataset.y_test, pred, average='binary')))
        #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "The Receiver Operating Characteristic Curve (ROC AUC): " + Fore.LIGHTGREEN_EX + str(roc_auc_score(dataset.y_test, pred)))
        print("\n")
        count+=1

    #count = 0
    #for classifier in classifiers:
        #clf = classifier
        #clf.fit(dataset.x_train_bin, dataset.y_train)
        #pred_bin = clf.predict(dataset.x_test_bin)
        #print(Fore.LIGHTCYAN_EX + names[count])
        #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(confusion_matrix(dataset.y_test, pred_bin)))
        #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(accuracy_score(dataset.y_test, pred_bin)))
        #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(precision_score(dataset.y_test, pred_bin, average='binary')))
        #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(recall_score(dataset.y_test, pred_bin, average='binary')))
        #print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "The Receiver Operating Characteristic Curve (ROC AUC): " + Fore.LIGHTGREEN_EX + str(roc_auc_score(dataset.y_test, pred_bin)))
        #count+=1

    go_bk_main(dataset)

def attack_summary_screen(dataset):
    banner()
    show_banner2("Evaluation summary on all classifiers")
    show_dataset(dataset)

    classifiers = [multiNomialNaiveBayesClassifier(), decisionTreeClassifier(), svmClassifier()]
    names = ["Naive Bayes Classifier:", "Decision Tree Classifier:", "Support Vector Machines Classifier:"]
    count = 0
    print(Fore.LIGHTMAGENTA_EX + "1- Good Word Attack : Best N Words" + "\n")
    for classifier in classifiers:
        clf = classifier
        clf.fit(dataset.x_train_bin, dataset.y_train)
        #pred_bin = clf.predict(dataset.x_test_bin)

        print(Fore.LIGHTCYAN_EX + names[count] + "\n")
        attack = GoodWordAttack(classifier=clf.model, attack_type="Best N Words")
        ad_set = attack.attack(test_set=dataset.x_test_bin, test_labels=dataset.y_test, malicious_set=None)
        ad_preds = clf.predict(ad_set)

        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(
            confusion_matrix(dataset.y_test, ad_preds, labels=[-1, 1])))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(
            accuracy_score(dataset.y_test, ad_preds)))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(
            precision_score(dataset.y_test, ad_preds, average='binary')))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(
            recall_score(dataset.y_test, ad_preds, average='binary')))
        print("\n")
        count+=1


    count = 0

    print(Fore.LIGHTMAGENTA_EX + "2- Good Word Attack : First N Words" + "\n")
    for classifier in classifiers:
        clf = classifier
        clf.fit(dataset.x_train_bin, dataset.y_train)
        #pred_bin = clf.predict(dataset.x_test_bin)

        print(Fore.LIGHTCYAN_EX + names[count] + "\n")
        attack = GoodWordAttack(classifier=clf.model, attack_type="First N Words")
        ad_set = attack.attack(test_set=dataset.x_test_bin, test_labels=dataset.y_test, malicious_set=None)
        ad_preds = clf.predict(ad_set)

        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(
            confusion_matrix(dataset.y_test, ad_preds, labels=[-1, 1])))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(
            accuracy_score(dataset.y_test, ad_preds)))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(
            precision_score(dataset.y_test, ad_preds, average='binary')))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(
            recall_score(dataset.y_test, ad_preds, average='binary')))
        print("\n")
        count+=1


    count = 0
    print(Fore.LIGHTMAGENTA_EX + "3- Feature Deletion Attack" + "\n")
    for classifier in classifiers:
        clf = classifier
        clf.fit(dataset.x_train_bin, dataset.y_train)
        pred_bin = clf.predict(dataset.x_test_bin)
        if count == 1:
                name = "DT"

        elif count == 0 :
                name = "MultinomialNB"
        else:
                name = "SVM"
        print(Fore.LIGHTCYAN_EX + names[count] + "\n")
        attack = FeatureDeletionAttack(clf.model, 1000, name)
        mal_set, mal_labels = attack.get_malicious_set(dataset.y_test, pred_bin, dataset.x_test_bin)
        ad_set = attack.attack(malicious_set=mal_set, test_labels=None, test_set=None)
        ad_preds = clf.predict(ad_set)

        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(
            confusion_matrix(mal_labels, ad_preds, labels=[-1, 1])))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(
            accuracy_score(mal_labels, ad_preds)))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(
            precision_score(mal_labels, ad_preds, average='binary')))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(
            recall_score(mal_labels, ad_preds, average='binary')))
        print("\n")
        count += 1

    count = 0
    print(Fore.LIGHTMAGENTA_EX + "4- Free Range Attack" + "\n")
    for classifier in classifiers:
        clf = classifier
        clf.fit(dataset.x_train, dataset.y_train)
        pred = clf.predict(dataset.x_test)

        print(Fore.LIGHTCYAN_EX + names[count] + "\n")


        attack = FreeRangeAttack(clf.model)
        mal_set, mal_labels = attack.get_malicious_set(dataset.y_test, pred, dataset.x_test)
        ad_set = attack.attack(test_set=dataset.x_test, malicious_set=mal_set, test_labels=None)
        ad_preds = clf.model.predict(ad_set)


        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(
            confusion_matrix(mal_labels, ad_preds, labels=[-1, 1])))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(
            accuracy_score(mal_labels, ad_preds)))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(
            precision_score(mal_labels, ad_preds, average='binary')))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(
            recall_score(mal_labels, ad_preds, average='binary')))
        print("\n")
        count += 1

    count = 0
    print(Fore.LIGHTMAGENTA_EX + "5- Restrained Attack" + "\n")
    for classifier in classifiers:
        clf = classifier
        clf.fit(dataset.x_train, dataset.y_train)
        pred = clf.predict(dataset.x_test)

        print(Fore.LIGHTCYAN_EX + names[count] + "\n")

        attack = RestrainedAttack(clf.model)
        mal_set, mal_labels = attack.get_malicious_set(dataset.y_test, pred, dataset.x_test)
        attack.set_innocuous_target(dataset.x_train, dataset.y_train, clf.model, "random")
        ad_set = attack.attack(malicious_set=mal_set, test_labels=None, test_set=None)
        ad_preds = clf.model.predict(ad_set)

        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Confusion Matrix: \n " + Fore.LIGHTGREEN_EX + str(
            confusion_matrix(mal_labels, ad_preds, labels=[-1, 1])))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Accuracy: " + Fore.LIGHTGREEN_EX + str(
            accuracy_score(mal_labels, ad_preds)))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Precision: " + Fore.LIGHTGREEN_EX + str(
            precision_score(mal_labels, ad_preds, average='binary')))
        print(Fore.LIGHTCYAN_EX + " (*) " + Fore.RESET + "Recall: " + Fore.LIGHTGREEN_EX + str(
            recall_score(mal_labels, ad_preds, average='binary')))
        print("\n")
        count += 1


    go_bk_main(dataset)

def init_app():
    e=screen1()
    screen2(e)

init_app()
