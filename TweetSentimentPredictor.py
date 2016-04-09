from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn import svm

from utils import get_random_state

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



def extract_sparse_matrix(documents, vectorizer):
    """
    Extract the document-term matrix from the documents.
    :param documents:
    :return:
    """
    X = vectorizer.transform(documents)
    return X

def result_string(clazz):
    if clazz == 0:
        return 'Positive'
    elif clazz == 1:
        return 'Negative'

    return ''

def words_from_tweets(tweets):
    s = set()
    for tweet in tweets:
        for word in tweet.split(" "):
            s.add(word)

    return s

######################### Script

DICTIONARY_FILE = '../WataProject/sentiment-dict.txt'
TWEETS_LABELLED_FILE = '../WataProject/training_data_file.csv'

words_from_dict = read_dictionary(DICTIONARY_FILE)
tweets, classes = read_tweets_labelled(TWEETS_LABELLED_FILE)

vectorizer = CountVectorizer(min_df=1, vocabulary=set(words_from_dict).union(words_from_tweets(tweets)), lowercase=True)

X = extract_sparse_matrix(tweets, vectorizer)
y = classes

# Classifier
clf = svm.SVC(kernel='linear', random_state=get_random_state())
# clf = svm.LinearSVC(random_state=get_random_state())
# clf = tree.DecisionTreeClassifier(random_state=get_random_state())
clf.fit(X, y)

X_test = ["accused of protests leave"]
y_test = clf.predict(extract_sparse_matrix(X_test, vectorizer))

print(y_test)
print("'" + X_test[0] + "'" + " is " + result_string(y_test[0]))