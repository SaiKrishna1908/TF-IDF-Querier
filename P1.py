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

documents = {}

# {token: [document1, document2]}
df_dict = {}

idf_dict = {}

# {token: frequency} in all collections of document
tf_dict = {}

wtd = {}

posting ={}

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
            
            content = read_file(fileName)
            tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
            tokens = stem_tokens(remove_stop_words(tokenizer.tokenize(content)))
            documents[fileName] = tokens

def compute_term_freq():
    for document, document_tokens in documents.items():
        tf_dict[document] = {}
        for token in document_tokens:
            tf_dict[document][token] = 1 if token not in tf_dict[document] else tf_dict[document][token] + 1

def compute_document_freq():
    for document, document_tokens in documents.items():
        for token in document_tokens:
            if token not in df_dict:
                df_dict[token] = [document]
            else:
                df_dict[token].append(document)

    for token in df_dict:
        df_dict[token] = set(df_dict[token])

def compute_log_tf():
    for (k,v) in tf_dict.items():
        for token in v:
          tf_dict[k][token] = (1 + math.log10(tf_dict[k][token]))    

def compute_inverse_df():
    for (k,v) in df_dict.items():
        idf_dict[k] = math.log10(40/len(v))


def compute_tf_idf():
    for (document, value) in tf_dict.items():
        wtd[document] = {}
        for token in value:
            wtd[document][token] = tf_dict[document][token] * idf_dict[token]            

def normalize_tf_idf():
    for (k,v) in wtd.items():
        scores =  v.values()
        sum = 0
                
        for score in scores:            
            sum = sum + (score**2)
                        
        sum = math.sqrt(sum)
        for (document_word, document_f) in v.items():
            wtd[k][document_word] = wtd[k][document_word] / sum
      
def getweight(documentName, token):
    stemmed_token = stemmer.stem(token)
    if documentName in wtd:
        if stemmed_token in wtd[documentName]:
            return wtd[documentName][stemmed_token]
        else:
            return 0
    return 'Invalid Document'

def process_query(search_term):
    search_term = stem_tokens(RegexpTokenizer(r'[a-zA-Z]+').tokenize(search_term))
    wtq = {}

    for item in search_term:
        wtq[item] = wtq[item] + 1 if item in wtq else 1
    
    sum = 0
    for (k,v) in wtq.items():
        wtq[k] = (1+math.log10(v))
        sum = sum + (wtq[k] ** 2)
    
    sum = math.sqrt(sum)
    for (k,v) in wtq.items():
        wtq[k] = v/sum

    return wtq

def query(search_term):
    wtq = process_query(search_term)
    candidate_documents = {}

    for token, query_weight in wtq.items():
        if token not in posting:
            return None,0

        top_documents = posting[token][:10]
        if len(top_documents) > 0:
            max_weight_in_posting = list(top_documents[0].values())[-1]
        else:
            max_weight_in_posting = 0

        for doc_info in top_documents:
            doc_name = list(doc_info.keys())[0]
            doc_weight = list(doc_info.values())[0]

            if doc_name not in candidate_documents:
                candidate_documents[doc_name] = {'actual_score': 0, 'upper_bound': 0}


            candidate_documents[doc_name]['actual_score'] += query_weight * doc_weight

        for doc_name in candidate_documents:
            if doc_name not in [list(d.keys())[0] for d in top_documents]:
                candidate_documents[doc_name]['upper_bound'] += query_weight * max_weight_in_posting

    best_doc = None
    best_score = -float('inf')
    is_pure = True

    for doc_name, scores in candidate_documents.items():

        total_score = scores['actual_score'] + scores['upper_bound']

        if total_score > best_score:
            if scores['upper_bound'] > 0:
                is_pure = False
            else:
                is_pure = True
            best_score = total_score
            best_doc = doc_name        
        

    if best_doc:
        if not is_pure:
            return ("fetch more", 0)
        return best_doc, best_score
    else:
        return None, 0


def getidf(term):
    tokens = RegexpTokenizer(r'[a-zA-Z]+').tokenize(term)
    tokens = stem_tokens(tokens)
    if tokens[0] in idf_dict:
        return idf_dict[tokens[0]]
    else:
        return 0

def get_tf(documentName, term):
    term = stem_tokens([term])[0]
    return tf_dict[documentName][term]


def construct_posting_list():
    for (document, v) in wtd.items():
        for (token, tf_idf) in v.items():
            if token not in posting:
                posting[token] = []
            
            posting[token].append({document: tf_idf})

    sort_posting()

def sort_posting():
    for token, document_tf_idf in posting.items():
        posting[token] = sorted(document_tf_idf, key=lambda x: list(x.values())[0], reverse=True)

populate_president_addressess()
compute_term_freq()
compute_document_freq()
compute_inverse_df()
compute_log_tf()
compute_tf_idf()
normalize_tf_idf()
construct_posting_list()

# print("%.12f" % getidf('democracy'))
# print("%.12f" % getidf('foreign'))
# print("%.12f" % getidf('states'))
# print("%.12f" % getidf('honor'))
# print("%.12f" % getidf('great'))
# print("--------------")
# print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
# print("%.12f" % getweight('23_hayes_1877.txt','public'))
# print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
# print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
# print("%.12f" % getweight('37_roosevelt_franklin_1933.txt','leadership'))
# print("--------------")
print("(%s, %.12f)" % query("states laws"))
# print("(%s, %.12f)" % query("war offenses"))
# print("(%s, %.12f)" % query("british war"))
# print("(%s, %.12f)" % query("texas government"))