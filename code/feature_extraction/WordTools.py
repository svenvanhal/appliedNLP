import os
import re
import sys
from itertools import chain
from PyDictionary import PyDictionary


class WordTools:
    """
    Process sentences and extract words.
    """

    # Regex pattern for all relevant punctuation
    dashpat = re.compile(r"-")
    doubledashpat = re.compile(r"--")
    puncpat = re.compile(r"[,;@#?!&$\"]+ *")

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

        # Catch empty sentences
        if obj is None:
            return -1

        # Get all (lowercased) words in string or list
        words = self.get_words(obj, unique=distinct)

        # Average the number of words in a list
        if not isinstance(obj, str):
            if len(obj) == 0: return -1
            return len(words) / len(obj)

        # Return the number of words in the sentence
        return len(words)

    def formal_words(self, distinct_words):
        """Returns a list of all formal words
        This method uses the PyDictionary lib. This plugin will search for the meaning of words on WordNet:
        http://wordnetweb.princeton.edu.

        Note:
            - There are two methods in PyDictionary for meanings - .meaning vs .googlemeaning
            - Meaning is the best as it uses the more strict WordNet dictionary. ex: iOS is not defined in WordNet but
            is in google dictionary
        """

        # PyDictionary will print errors if the word was not found
        # We are going to ignore that by changing stdout
        sys.stdout = open(os.devnull, 'w')

        # Search for all words simultaneously
        pydict = PyDictionary(distinct_words)
        meanings = pydict.getMeanings()

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Formal words have a meaning
        formal = list(k for k, v in meanings.items() if v is not None)

        # TODO: do we also need the informal words?
        # informal = list(k for k, v in meanings.items() if v is None)

        return formal

    def __get_all_words(self, obj) -> list:
        """Return all words in a string or iterable of strings."""

        # Catch empty sentences
        if obj is None:
            return []

        # Catch non-strings (probably list / pd.Series)
        if not isinstance(obj, str):

            # Catch empty lists
            if len(obj) == 0:
                return []

            # Convert each sentence to words
            words = map(self.__get_all_words, obj)

            # Flatten result [["a", "b"], ["c"]] to ["a", "b", "c"]
            return list(chain.from_iterable(words))

        # Replace punctuation with space, convert to lowercase
        obj = self.__clean(obj)

        return obj.split()

    def __clean(self, obj: str) -> str:
        """Replace punctuation with a space, convert to lowercase."""

        # First replace double dash with space
        obj = self.doubledashpat.sub(" ", obj)

        # Then replace single dash with nothing
        obj = self.dashpat.sub("", obj)

        # Then replace the remaining punctuation
        return self.puncpat.sub(" ", obj).lower()
