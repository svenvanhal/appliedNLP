import os
import re
import sys
from PyDictionary import PyDictionary


class WordTools:
    """
    Process sentences and extract words.
    """

    # Regex pattern for all relevant punctuation
    dashpat = re.compile(r"-")
    doubledashpat = re.compile(r"--")
    puncpat = re.compile(r"[,;@#?!&$\"]+ *")

    def count_words(self, obj):
        """Return the word count of a string."""

        # Catch empty sentences
        if obj is None:
            return -1

        # Catch non-strings (probably list / pd.Series)
        if not isinstance(obj, str):

            # Catch empty lists
            if len(obj) == 0:
                return -1

            # Average the number of words in a list
            return sum(map(self.count_words, obj)) / len(obj)

        return len(self.__get_words(obj))

    def __get_words(self, obj):
        """Return all words in a string."""

        # Replace punctuation with space
        obj = self.__clean(obj)

        return obj.split()

    def __get_words_from_list(self, obj):
        """Return all words in a list of strings"""

        # Check if it is really a list
        if isinstance(obj, list):
            # Catch empty lists
            if len(obj) == 0:
                return []

            # Extract all words and return one list
            words = []
            for w in obj:
                words.extend(self.__get_words(w))

            return words
        else:
            raise TypeError("List was expected")

    def __clean(self, obj):
        """Replace punctuation with a space."""

        # First replace double dash with space
        obj = self.doubledashpat.sub(" ", obj)

        # Then replace single dash with nothing
        obj = self.dashpat.sub("", obj)

        # Then replace the remaining punctuation
        return self.puncpat.sub(" ", obj)

    def get_distinct_words(self, obj):

        # Catch non-strings (probably list / pd.Series)
        if not isinstance(obj, str):
            words = self.__get_words_from_list(obj)
        else:
            words = self.__get_words(obj)

        # Cast to set to make distinct set
        distinct = list(set(words))

        return distinct

    def formal(self, distinct_words):
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
