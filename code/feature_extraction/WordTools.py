import re

from itertools import chain

from nltk import download
from nltk.data import find
from nltk.corpus import wordnet as wn


class WordTools:
    """
    Processes sentences to extract and count words.
    """

    # Regex pattern for all relevant punctuation
    dashpat = re.compile(r"-")
    doubledashpat = re.compile(r"--")  # TODO: 2 or more dashes
    puncpat = re.compile(r"[,;@#?!&$\"]+ *")

    def __init__(self):

        # Download the WordNet corpus if not installed
        try:
            find('corpora/wordnet.zip')
        except LookupError:
            download('wordnet')

    def get_words(self, obj, unique=False) -> list:
        """
        Get either all words in a string (or iterable of strings), or only the unique words.
        All words are converted to lowercase and punctuation is stripped.
        Returns an empty list for invalid arguments or empty input arguments.
        """

        # Get all words
        words = self.__get_all_words(obj)

        # Return either only unique words or all words
        return list(set(words)) if unique else words

    def count_words(self, obj, distinct=False):
        """
        Return the word count of a string.
        Wrapper around len(words) to accommodate lists of words (for which the count is averaged)
        Returns -1 for empty lists or non-existent strings. 0 for empty strings.
        """

        # Catch empty sentences (also catches empty list)
        if not obj and obj != "":
            return -1

        # Get all (lowercased) words in string or list
        words = self.get_words(obj, unique=distinct)

        # Average the number of words in a list
        if not isinstance(obj, str):
            return len(words) / len(obj)

        # Return the number of words in the sentence
        return len(words)

    def formal_words(self, distinct_words):
        """
        Returns a list of all formal words, using WordNet via NLTK.
        N.B.: Is case-sensitive!
        """

        # Get the set of all words which exist in the WordNet corpus
        return {word for word in distinct_words if wn.synsets(word)}

    def __get_all_words(self, obj) -> list:
        """
        Return all words in a string or iterable of strings.
        """

        # Catch empty sentences
        if not obj and obj != "":
            return []

        # Catch non-strings (probably list / pd.Series)
        if not isinstance(obj, str):
            # Convert each sentence to words
            words = map(self.__get_all_words, obj)

            # Flatten result [["a", "b"], ["c"]] to ["a", "b", "c"]
            return list(chain.from_iterable(words))

        # Replace punctuation with space, convert to lowercase
        obj = self.__clean(obj)

        return obj.split()

    def __clean(self, obj: str) -> str:
        """
        Replace punctuation with a space and converts input to lowercase.
        """

        # First replace double dash with space
        obj = self.doubledashpat.sub(" ", obj)

        # Then replace single dash with nothing
        obj = self.dashpat.sub("", obj)

        # Then replace the remaining punctuation
        return self.puncpat.sub(" ", obj).lower()
