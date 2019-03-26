import re


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

    def __clean(self, obj):
        """Replace punctuation with a space."""

        # First replace double dash with space
        obj = self.doubledashpat.sub(" ", obj)

        # Then replace single dash with nothing
        obj = self.dashpat.sub("", obj)

        # Then replace the remaining punctuation
        return self.puncpat.sub(" ", obj)
