import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
short_pos = open("positive.txt","r").read()  
short_neg = open("negative.txt","r").read()  

all_words = []
documents = []


#  j is adject, r is adverb, and v is verb
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p.decode('latin-1'))   
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p.decode('latin-1'))  
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]


save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document.decode('latin-1'))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets = open("pickled_algos/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

### to suppress numpy warnings
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
    
# OUTPUT    
# dhcp-10-202-152-177:Sentiment Neo$ python Pickling_classifiers.py 
# 10664
# ('Original Naive Bayes Algo accuracy percent:', 70.33132530120481)
# Most Informative Features
#                 mediocre = True              neg : pos    =     15.7 : 1.0
#               refreshing = True              pos : neg    =     13.6 : 1.0
#                wonderful = True              pos : neg    =     12.6 : 1.0
#                     warm = True              pos : neg    =     12.2 : 1.0
#              mesmerizing = True              pos : neg    =     11.6 : 1.0
#                   stupid = True              neg : pos    =     11.0 : 1.0
#                     dull = True              neg : pos    =     10.9 : 1.0
#                     thin = True              neg : pos    =     10.6 : 1.0
#                 powerful = True              pos : neg    =     10.1 : 1.0
#                 tiresome = True              neg : pos    =      9.7 : 1.0
#                offensive = True              neg : pos    =      9.7 : 1.0
#                      wry = True              pos : neg    =      9.6 : 1.0
#                 supposed = True              neg : pos    =      9.4 : 1.0
#                     loud = True              neg : pos    =      9.0 : 1.0
#                  harvard = True              neg : pos    =      9.0 : 1.0

