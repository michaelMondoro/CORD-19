"""
TASK:
What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?

Specifically, we want to know what the literature reports about:

    Data on potential risks factors
        - Smoking, pre-existing pulmonary disease
        - Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities
        - Neonates and pregnant women
        - **Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
    
    - Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
    - Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
    - **Susceptibility of populations
    - **Public health mitigation measures that could be effective for control


version 3.17.2020
"""
import pandas
import json
import random
import os
from shutil import copyfile
from pprint import pprint
import nltk
from nltk.corpus import stopwords

'''
*DataFrame*
paper_df_sections(['paper_id', 'abstract', 'body_text', 'back_matter', 'metadata.title',
       'metadata.authors', 'bib_entries ...', 'ref_entries ...'])

*DataFrame.Series*
body_text/abstract/back_matter(['text', 'cite_spans', 'ref_spans', 'section'])

'''

# Load a single file and returns a DataFrame 
#
# file - filename of file to open
def load_data(file):
    # Open sample file 
    f = open(file)

    # Load json data
    data = json.load(f)
    # Load json into pandas DataFrame
    data = pandas.json_normalize(data)
    return data


# Print information about the given DataFrame
#
# data - pandas DataFrame
def data_stats(filename, data):
    print("Filename: {}".format(filename))
    print("Number of Sections:")
    print("\tabstract: {}".format(len(data['abstract'][0])))
    print("\tbody_text: {} ".format(len(data['body_text'][0])))

    mentions = covid_mentions(data)
    print("Keyword mentions: {}".format(mentions))

# Load the given number of files from the 'comm_use_subset' directory.
# Returns list of tuples (filename, DataFrame)
def load_multiple(n, start_index=0):
    loaded_list = []
    # Read files from directory
    file_list = os.listdir("comm_use_subset")
    i = start_index
    while i < n:
        data = load_data("comm_use_subset/" + file_list[i])
        loaded_list.append((file_list[i], data))
        i += 1

    return loaded_list


# Returns a tuple of keyword mentions in the given DataFrame and a list of InfoPackets
def covid_mentions(filename, data):
    mentions = 0
    for section in data:
        if section == 'abstract' or section == 'body_text':
            for entry in data[section][0]:
                text = entry['text'].lower()
                # print(entry['text'])
                # print("####")
                if "coronavirus" in text or "covid-19" in text or "hcov" in text:
                    mentions += 1

    return mentions


# Returns a list of tuples. Only includes files with keywords > num_mentions
#
# num_files - number of files to load
# num_mentions - min number of keywords to make file relevant
def find_relevant(num_files, num_mentions, start_index=0):
    files = load_multiple(num_files)
    relevant_files = []

    for file in files:
        mentions = covid_mentions(file[0], file[1])
        if mentions > num_mentions:
            relevant_files.append(file)
    return relevant_files



'''
Class representing a packet of relevant information.

    filename - name of the file from which the info was found
    df - DataFrame object containing the info
    df_section - section of the DataFrame where the info can be found
    keywords - list of any keywords associated with this info
    data - string of raw data itself

'''
class InfoPacket:
    def __init__(self, filename, df, data, df_section=None, keywords=[]):
        self.filename = filename
        self.df = df
        self.df_section = df_section
        self.keywords = keywords
        self.data = data
        self.packet_id = filename[0:5] + str(random.randrange(0,100))
    

    def __str__(self):
        return "--InfoPacket--\n[Filename]: {}  [DataFrame Section]: {}  [Keywords]: {}\n[Data]: \n{}\n---\n".format(self.filename, self.df_section, self.keywords, self.data)


