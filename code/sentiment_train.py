import pandas as pd   
from bs4 import BeautifulSoup 
import re
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.porter import PorterStemmer

def clean_review(raw_review):
	example1 = BeautifulSoup(raw_review, 'html.parser').get_text()
	# print type(example1)
	# Use regular expressions to do a find-and-replace
	letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
	                      " ",                   # The pattern to replace it with
	                      example1 )  # The text to search
	# print letters_only
	lower_case = letters_only.lower()        # Convert to lower case
	words = lower_case.split()               # Split into words
	# print words
	# print stopwords.words("english")
	words = [w for w in words if not w in stopwords.words("english")]
	# print words
	from nltk.stem.porter import PorterStemmer
	porter_stemmer = PorterStemmer()
	stemmed_words = []
	for i in range(len(words)):
		stemmed_words.append(porter_stemmer.stem(words[i]))
	return( " ".join( stemmed_words ))


train = pd.read_csv("C:/Users/Ash/Documents/GitHub/nlp_bow/data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train.shape)
# print train.columns.values
num_reviews = train["review"].size
clean_reviews = []
for i in range(num_reviews):
	if( (i+1)%1000 == 0 ):
		print "Review %d of %d\n" % ( i+1, num_reviews )                                                          
	clean_reviews.append(clean_review(train["review"][i]))

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )


