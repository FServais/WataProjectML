from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree, ensemble
from sklearn import svm, cross_validation
from sklearn.cross_validation import KFold
from sklearn import grid_search

import numpy as np

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
    return list(
        map(lambda splittedLine: splittedLine[2][len("word1="):], lines))  # Extract the word (following "word1=")


def read_tweets_labelled(filePath):
    """ Read file containing tweets and their correspinding class.
    Note: Format of a line of the file: <TWEET>,<CLASS>, where <CLASS> is either 0 (positive) or 1 (negative)
    :param filePath: Path to the file.
    :return: Tuple with a list of string of the tweets, and a list of the corresponding classes
    """
    with open(filePath) as f:
        content = f.readlines()

    lines = [x.rstrip('\n').split(",") for x in content]

    lines_cleaned = []
    for p in lines:
        if len(p) > 1:
            lines_cleaned.append(p)

    tweets = [line[0] for line in lines_cleaned]
    classes = [int(line[1]) for line in lines_cleaned]

    return np.array(tweets), np.array(classes)


def extract_sparse_matrix(documents, vectorizer):
    """
    Extract the document-term matrix from the documents.
    :param documents:
    :return:
    """
    X = vectorizer.transform(documents)
    return X


def result_string(clazz):
    if clazz == 1:
        return 'Positive'
    elif clazz == -1:
        return 'Negative'
    else:
        return 'Neutral'



def words_from_tweets(tweets):
    s = set()
    for tweet in tweets:
        for word in tweet.split(" "):
            s.add(word)

    return s


######################### Script

DICTIONARY_FILE = '../WataProject/sentiment-dict.txt'
TWEETS_LABELLED_FILE = '../WataProject/training_data_file.csv'

N_FOLDS = 10

words_from_dict = read_dictionary(DICTIONARY_FILE)
tweets, classes = read_tweets_labelled(TWEETS_LABELLED_FILE)

vectorizer = CountVectorizer(min_df=1, vocabulary=set(words_from_dict).union(words_from_tweets(tweets)), lowercase=True)

# Select classifier
clf = svm.LinearSVC(random_state=get_random_state())
parameters = {'C': np.arange(0.01, 1.05, 0.05), 'loss': ['hinge', 'squared_hinge']}

# clf = tree.DecisionTreeClassifier(random_state=get_random_state())
# parameters = {'max_depth': np.arange(1, 2000, 50)}

# clf = svm.SVC(random_state=get_random_state())
# parameters = {'C': np.arange(0.1, 1.1, 0.1), 'kernel': ['poly', 'rbf'], 'gamma': np.arange(0.00005, 0.00006, 0.000001)}

# clf = ensemble.ExtraTreesClassifier(random_state=get_random_state())
# parameters = {'criterion': ['gini']}


new_tweets = ["New episode is awesome #GoTSeason6", "Best show ever #GoTSeason6 #ValarMorghulis", "Not that bad #GoTSeason6", "So disappointed, so slow #GoTSeason6", "Stopping watching the show, nothing like the books #GoTSeason6"]

############ Model evaluation
print("Model evaluation")
print("--------------------")
k_fold = KFold(n=len(classes), n_folds=N_FOLDS)

n_correct_tot = 0
n_incorrect_tot = 0

for train_indices, test_indices in k_fold:
    X_train = extract_sparse_matrix(tweets[train_indices], vectorizer)
    y_train = classes[train_indices]

    X_test = tweets[test_indices]
    y_test = classes[test_indices]

    # Classifier
    clf.fit(X_train, y_train)

    y_out = clf.predict(extract_sparse_matrix(X_test, vectorizer))

    assert(len(y_out) == len(y_test))
    result = (y_out == y_test)
    n_correct = len(result[result == True])
    n_incorrect = len(result[result == False])

    n_correct_tot += n_correct
    n_incorrect_tot += n_incorrect

print("Well classified      : {:.2f}%".format(n_correct_tot * 100 / (n_correct_tot + n_incorrect_tot)))
print("Misclassified        : {:.2f}%".format(n_incorrect_tot * 100 / (n_correct_tot + n_incorrect_tot)))
print("Using cross_val_score: {:.4f}".format(cross_validation.cross_val_score(clf, extract_sparse_matrix(tweets, vectorizer), classes, cv=N_FOLDS).mean()))

#### New tweets
X_new_train = extract_sparse_matrix(tweets, vectorizer)
y_new_train = classes
clf.fit(X_new_train, y_new_train)
y_new_out = clf.predict(extract_sparse_matrix(new_tweets, vectorizer))
y_new_out_str = list(map(result_string, y_new_out))

for i in range(0, len(new_tweets)):
    print("Tweet: '" + new_tweets[i] + "' is " + y_new_out_str[i])


############ Compute best parameters
grid = grid_search.GridSearchCV(estimator=clf, param_grid=parameters, cv=N_FOLDS)
grid.fit(extract_sparse_matrix(tweets, vectorizer), classes)


clf = grid.best_estimator_

print('\n')
############ New model evaluation
print("New model evaluation")
print("--------------------")
n_correct_tot = 0
n_incorrect_tot = 0

for train_indices, test_indices in k_fold:
    X_train = extract_sparse_matrix(tweets[train_indices], vectorizer)
    y_train = classes[train_indices]

    X_test = tweets[test_indices]
    y_test = classes[test_indices]

    # Classifier
    clf.fit(X_train, y_train)

    y_out = clf.predict(extract_sparse_matrix(X_test, vectorizer))

    result = (y_out == y_test)
    n_correct = len(result[result == True])
    n_incorrect = len(result[result == False])

    n_correct_tot += n_correct
    n_incorrect_tot += n_incorrect

print("Well classified      : {:.2f}%".format(n_correct_tot * 100 / (n_correct_tot + n_incorrect_tot)))
print("Misclassified        : {:.2f}%".format(n_incorrect_tot * 100 / (n_correct_tot + n_incorrect_tot)))
print("Using cross_val_score: {:.4f}".format(cross_validation.cross_val_score(grid.best_estimator_, extract_sparse_matrix(tweets, vectorizer), classes, cv=N_FOLDS).mean()))

#### New tweets
X_new_train = extract_sparse_matrix(tweets, vectorizer)
y_new_train = classes
clf.fit(X_new_train, y_new_train)
y_new_out = clf.predict(extract_sparse_matrix(new_tweets, vectorizer))
y_new_out_str = list(map(result_string, y_new_out))

for i in range(0, len(new_tweets)):
    print("Tweet: '" + new_tweets[i] + "' is " + y_new_out_str[i])