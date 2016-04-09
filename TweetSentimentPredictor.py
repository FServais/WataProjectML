def read_dictionary(filePath):
    # Read dictionary of words and extract all the words in it.
    # Format of a line of the dictionary: 'type=... len=... word1=<WORD EXTRACTED> pos1=... stemmed1=... priorpolarity=...'
    # Returns a list of words
    with open(filePath) as f:
        content = f.readlines()

    lines = [x.rstrip('\n').split(" ") for x in content]
    return list(map(lambda splittedLine: splittedLine[2][len("word1="):], lines)) # Extract the word (following "word1=")


def read_tweets_labelled(filePath):
    # Read file containing tweets and their correspinding class.
    # Format of a line of the file: <TWEET>,<CLASS>, where <CLASS> is either 0 (positive) or 1 (negative)
    # Returns a dict() with the tweet as the key and the class as value.
    with open(filePath) as f:
        content = f.readlines()

    lines = [x.rstrip('\n').split(",") for x in content]
    listOfDict = list(map(lambda splittedLine: {splittedLine[0]: splittedLine[1]}, lines))

    toReturn = {}
    for d in listOfDict:
        toReturn.update(d)

    return toReturn


######################### Script

DICTIONARY_FILE = '../WataProject/sentiment-dict.txt'
TWEETS_LABELLED_FILE = 'test_ml.txt'

words_from_dict = read_dictionary(DICTIONARY_FILE)
tweets_labelled = read_tweets_labelled(TWEETS_LABELLED_FILE)



