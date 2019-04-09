class Util:

    @staticmethod
    def ratio(left, right, raw=False):
        """
        Returns the ratio between two numbers.
        If any argument is undefined or <= 0, returns -1.
        """

        # Catch edge cases
        if not left or left <= 0 or not right or right <= 0:
            return -1 if not raw else 0

        return abs(left / right)

    @staticmethod
    def diff(left, right, raw=False):
        """
        Return the difference between two numbers.
        If any argument is undefined or < 0, returns -1.
        """

        # Catch edge cases (check that both sides "exist")
        if not left or left < 0 or not right or right < 0:
            return -1 if not raw else 0

        return abs(left - right)

    @staticmethod
    def ratio_raw(left, right):
        return Util.ratio(left, right, True)

    @staticmethod
    def diff_raw(left, right):
        return Util.diff(left, right, True)

    @staticmethod
    def count_chars(obj):
        """
        Determines the number of characters in a string.
        If the argument is undefined or an empty list, returns -1.
        """

        # Catch empty post titles (and empty lists)
        if not obj:
            return -1

        # Catch non-strings (probably list / pd.Series)
        if not isinstance(obj, str):
            # Average the length of items in a list
            return sum(map(Util.count_chars, obj)) / len(obj)

        # Strip spaces
        processed = obj.strip().replace(" ", "")

        # Return string length
        return len(processed)

    @staticmethod
    def count_specific_char(obj, char):
        """
        Counts the number of occurrences of a specific (sub)string.
        If any arguments are undefined, or when the (sub)string is not found, returns 0.
        """

        if not char or not obj:
            return 0

        # Catch non-strings (probably list / pd.Series)
        if not isinstance(obj, str):
            # Sum the number of occurrences in each list item
            return sum([Util.count_specific_char(item, char) for item in obj])

        return obj.count(char)

    @staticmethod
    def count_words(obj):
        """
        Determines the number of words in a list.
        If the argument is undefined or an empty list, returns -1.
        """

        # Catch empty input args
        if not obj:
            return -1

        # Catch nested lists
        if isinstance(obj[0], list):

            # Catch empty lists
            if not obj[0]: return -1

            # Average the length of items in a list
            return sum(map(Util.count_words, obj)) / len(obj)

        # Return string length
        return len(obj)

    @staticmethod
    def count_words_titlecase(obj):

        # Catch empty post titles (and empty lists)
        if not obj: return -1

        # Catch nested lists
        if isinstance(obj[0], list):

            # Catch empty lists
            if not obj[0]: return -1

            # Average the length of items in a list
            return sum(map(Util.count_words_titlecase, obj)) / len(obj)

        words = 0
        for word in obj:
            if word[0].isupper(): words += 1

        # Return -1 if no uppercase words
        return words if words > 0 else -1

    @staticmethod
    def count_words_uppercase(obj):
        # TODO: merge with count words and count titlecase

        # Catch empty post titles (and empty lists)
        if not obj: return -1

        # Catch nested lists
        if isinstance(obj[0], list):

            # Catch empty lists
            if not obj[0]: return -1

            # Average the length of items in a list
            return sum(map(Util.count_words_uppercase, obj)) / len(obj)

        words = 0
        for word in obj:
            if word.isupper(): words += 1

        # Return -1 if no uppercase words
        return words if words > 0 else -1

    @staticmethod
    def count_tags(obj, tags: set) -> float:
        """
        Counts the number of words with a tag in the provided tag set.
        """

        # Catch empty post titles (and empty lists)
        if not obj:
            return 0

        # Catch non-strings (probably list / pd.Series)
        if isinstance(obj[0], list):
            # Average the length of items in a list
            return sum(map(lambda x: Util.count_tags(x, tags), obj)) / len(obj)

        return sum([1 for pos_tuple in obj if pos_tuple[1] in tags])

    @staticmethod
    def is_retweet(obj) -> int:

        if not isinstance(obj, str):
            return 0

        return 1 if obj[:3] == 'RT ' else 0
