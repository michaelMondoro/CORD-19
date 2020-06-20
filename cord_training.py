""""
    File for compiling training data.

    TODO: Remove stopwords
version 3.19.2020
"""

import pandas
import json
import re
import os
import numpy as np
import nltk
import sqlite3
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from covid_analysis import *

# Keyword list
risk_factor_keywords = ["smoking", "co-infection", "pregnant", "pulmonary disease", 
                        "risk factors", "pre-existing", "neonates", "transmissible", 
                        "infections", "co-morbidities", "severity", "disease", "fatal", "symptomatic", "high-risk"]

class_dict = {  'smoking': ["pulmonary disease", "smok"], 
                'co-infection': ["co-infection", "pre-exist", "infection", "co-morbidities", "coinfection", "co-exist"], 
                'pregnant': ["pregnant", "neonates"], 
                'transmission': ["transmissible", "transmission", "carry", "reproduc", "incubat", "serial interval", "transfer"], 
                'severity': ["severity", "fatal", "symptomatic", "high-risk", "death", "hospit", "ICU", "extreme"]}


out_file = open("training/test_data.txt")
db = sqlite3.connect("training_database.db")
cursor = db.cursor()


def generate_packet(filename, data):
    info = None
    paragraphs = []

    for section in data:
        if section == 'abstract' or section == 'body_text' or section == 'back_matter':
            for entry in data[section][0]:
                text = entry['text'].lower()
                # print(entry['text'])
                # print("####")
                paragraphs.append(text)
                paragraphs.pop(0) if len(paragraphs) > 3 else paragraphs == paragraphs

                if "coronavirus" in text or "hcov" in text or "covid-19" in text:
                    combined_text = ' '.join(map(str, paragraphs))
                    keywords = check_keywords(combined_text)
                    info = InfoPacket(filename, data, combined_text, df_section=section, keywords=keywords)

    return info

# Generate list of InfoPackets from the given list of files
def generate_packets(relevant_files):
    info_packets = []
    for file in relevant_files:
        info = generate_packet(file[0], file[1])
        info_packets.append(info)

    return info_packets

# Get the number of keywords in the given data string
def check_keywords(data):
    keywords = []
    for classification in class_dict:
        for word in class_dict[classification]:
            if (word in data) and (word not in keywords):
                keywords.append(word)
            else:
                keywords = keywords

    return keywords

# Lambda to sort packets by their number of associated keywords
def keyword_lambda(packet):
    return len(packet.keywords)

# Write packet list to database
def write_to_db(packet_list):
    cursor.execute('SELECT filename FROM {} '.format(packets))
    filenames = cursor.fetchall()

    for packet in packet_list:
        keywords = ""
        for word in packet.keywords:
            keywords = keywords + " " + word
        cursor.execute("""INSERT INTO packets (filename, data, keywords, class) VALUES (?,?,?,?)""", (packet.filename, packet.data, keywords, ""))

    db.commit()


# Loop through given number of files and extract relevant snippets from the text.
#
# Write output to test_data.txt
def gather_data(n):
    relevant_files = find_relevant(n, 3, start_index=25)
    packets = generate_packets(relevant_files)
    #packets.sort(key=keyword_lambda, reverse=True)
    write_to_db(packets)

def clean(string):
    words = []
    for s in re.split(r"[,']", string):
        if len(s) > 1:
            words.append(s)
    return words


# Automatically label data based on keyword mentions
#
# Return list of tuples (InfoPacket, classification)
def label(num_files):
    labels = []
    
    print("Finding relevant files...")
    relevant = find_relevant(num_files, 3, start_index=25)
    print("Generating packets...")
    packets = generate_packets(relevant)
    i = 0
    for packet in packets:
        counts = {'smoking':0,'co-infection':0,'pregnant':0,'transmission':0,'severity':0}
        for classification in class_dict:
            for word in classification:
                if word in packet.data:
                    counts[classification] += 1 

        c = get_class(counts)  
        labels.append((packet, c))
        cursor.execute("""INSERT INTO auto_labelled (data, class) VALUES (?,?)""", (packet.data, c))
        i += 1
    db.commit()
    return labels

# Finds the class that has the most keyword references
def get_class(dictionary):
    classes = list(dictionary.keys())
    values = list(dictionary.values())
    result = classes[values.index(max(values))]
    return result



# Read data from database. Raw string data and classification label
def read_data(table_name):
    packets = []
    cursor.execute('SELECT * FROM {} '.format(table_name))

    rows = cursor.fetchall()

    if table_name == "packets":
        for row in rows:
            packets.append((row[1], row[3]))
    else:
        for row in rows:
            packets.append((row[0], row[1]))

    return packets


def train_and_predict():
    ''' 
        Training sequence:      
    '''
    # 1) Read labelled data from database
    # packets_read = read_labelled_data()
    print("Reading data...")
    #packets_read = label(3000)
    packets = read_data("packets")
    auto_labelled = read_data("auto_labelled")

    # 2) Create array with all text from InfoPackets
    # 3) Create classification array (y)
    print("Formatting data...")
    words = []
    y = []


    for packet in packets:
        words.append(packet[0])
        y.append(packet[1])


    # 4) Create classifier and vectorizor
    clf = MultinomialNB()
    cv = TfidfVectorizer() 

    # 5) Make classification nparray (y) and training data nparray (X)
    y = np.asarray(y)
    X = cv.fit_transform(words).toarray()

    # 6) Split data. Train and Predict. 
    print("Training...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.29, random_state=3)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    pred = clf.predict(X_test)

    #pickle.dump(clf, open('covid_model.sav', 'wb'))


    print("Results: \n{}\nScore: {}".format(pred, score))
    print("Expected: \n{}".format(y_test))


#label(100)
#train_and_predict()
#gather_data(1000)

'''
print("Finding relevant files...")
relevant = find_relevant(num_files, 3, start_index=25)
print("Generating packets...")
packets = generate_packets(relevant)

X = []
for packet in packets:
    X.append(packet[0])


kmeans = KMeans(n_clusters=5)
'''


