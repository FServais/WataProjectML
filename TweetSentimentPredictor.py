from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

def read_dictionary(filePath):
    """ Read dictionary of words and extract all the words in it.
    Note: Format of a line of the dictionary: 'type=... len=... word1=<WORD EXTRACTED> pos1=... stemmed1=... priorpolarity=...'
    :param filePath: Path to the dictionary.
    :return: list of words of the dictionary.
    """
    with open(filePath) as f:
        content = f.readlines()

    lines = [x.rstrip('\n').split(" ") for x in content]
    return list(map(lambda splittedLine: splittedLine[2][len("word1="):], lines)) # Extract the word (following "word1=")


def read_tweets_labelled(filePath):
    """ Read file containing tweets and their correspinding class.
    Note: Format of a line of the file: <TWEET>,<CLASS>, where <CLASS> is either 0 (positive) or 1 (negative)
    :param filePath: Path to the file.
    :return: Tuple with a list of string of the tweets, and a list of the corresponding classes
    """
    with open(filePath) as f:
        content = f.readlines()

    lines = [x.rstrip('\n').split(",") for x in content]

    tweets = [line[0] for line in lines]
    classes = [int(line[1]) for line in lines]

    return tweets, classes



def extract_sparse_matrix(tweets):
    vectorizer = CountVectorizer(min_df=1)

    X = vectorizer.fit_transform(tweets)
    return X, vectorizer

def extract_new_line(line):
    return vectorizer.transform(line)

def result_string(clazz):
    if clazz == 0:
        return 'Positive'
    elif clazz == 1:
        return 'Negative'

    return ''

######################### Script

DICTIONARY_FILE = '../WataProject/sentiment-dict.txt'
TWEETS_LABELLED_FILE = 'test_ml.txt'

words_from_dict = read_dictionary(DICTIONARY_FILE)
tweets, classes = read_tweets_labelled(TWEETS_LABELLED_FILE)
X, vectorizer = extract_sparse_matrix(tweets)
y = classes

# Classifier
clf = svm.SVC()
clf.fit(X, y)

X_test = ["Mike is wonderful"]
y_test = clf.predict(extract_new_line(X_test))

print("'" + X_test[0] + "'" + " is " + result_string(y_test[0]))