import os
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math

president_addressess_map = {}

DATA_ROOT = 'US_Inaugural_Addresses'

english_stop_words = stopwords.words('english')

stemmer = PorterStemmer()

df_dict = {}

tf_dict = {}

wtd = {}

def get_president_name(fileName):
    return fileName.split('_')[1]

def read_file(fileName):
    file = open(os.path.join(DATA_ROOT, fileName), 'r', encoding='windows-1252')
    doc = file.read()
    file.close()
    return doc.lower()

def remove_stop_words(tokens):
    
    final_tokens = []
    for token in tokens:
        if token not in english_stop_words:
            final_tokens.append(token)
    return final_tokens

def stem_tokens(tokens):
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    
    return stemmed_tokens

def populate_president_addressess():
    for fileName in os.listdir(DATA_ROOT):
        if not fileName.startswith('.'):
            presidentName = get_president_name(fileName)
            
            content = read_file(fileName)
            tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
            tokens = stem_tokens(remove_stop_words(tokenizer.tokenize(content)))
            tf_dict[fileName] = {}
            for token in set(tokens):
                df_dict[token] = 1 if token not in df_dict else df_dict[token] + 1

            for token in tokens:
                tf_dict[fileName][token] = 1 if token not in tf_dict[fileName] else tf_dict[fileName][token] + 1

            # if  presidentName in president_addressess_map:
            #     president_addressess_map[presidentName].append(tokens)
            # else:
            #     president_addressess_map[presidentName] = [tokens]

def compute_tf_idf():
    for (k,v) in tf_dict.items():
        tokens = v.keys()
        wtd[k] = {}
        for token in tokens:
            wtd[k][token] = (1 + math.log10(tf_dict[k][token])) * (math.log10(40 / df_dict[token]))

populate_president_addressess()
compute_tf_idf()

print(wtd)