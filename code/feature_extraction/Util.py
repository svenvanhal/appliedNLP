class Util:

    @staticmethod
    def ratio(left, right, raw=False):
        """
        Returns the ratio between two numbers.
        If any argument is undefined or <= 0, returns 0.
        """

        # Catch edge cases
        if not left or left <= 0 or not right or right <= 0:
            return 0 if not raw else 0

        return abs(left / right)

    @staticmethod
    def diff(left, right, raw=False):
        """
        Return the difference between two numbers.
        If any argument is undefined or < 0, returns 0.
        """

        # Catch edge cases (check that both sides "exist")
        if not left or left < 0 or not right or right < 0:
            return 0 if not raw else 0

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
        If the argument is undefined or an empty list, returns 0.
        """

        # Catch empty post titles (and empty lists)
        if not obj:
            return 0

        # Catch non-strings (probably list / pd.Series)
        if not isinstance(obj, str):
            # Average the length of items in a list
            return sum(map(Util.count_chars, obj)) / len(obj)

        # Strip spaces
        processed = obj.strip().replace(" ", "")

        # Return string length (cap at 300 characters)
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
        If the argument is undefined or an empty list, returns 0.
        """

        # Catch empty input args
        if not obj:
            return 0

        # Catch nested lists
        if isinstance(obj[0], list):

            # Catch empty lists
            if not obj[0]: return 0

            # Average the length of items in a list
            return sum(map(Util.count_words, obj)) / len(obj)

        # Return string length
        return len(obj)

    @staticmethod
    def count_words_case(obj):
        """Returns (num_titlecase, num_uppercase)"""

        # Catch empty post titles (and empty lists)
        if not obj:
            return 0, 0

        tc = 0
        uc = 0
        for word in obj:
            if word.istitle():
                tc += 1
            elif word.isupper():
                uc += 1

        if tc <= 0: tc = 0
        if uc <= 0: uc = 0

        # Return 0 if no uppercase words
        return tc, uc

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

        num_tags = sum([1 for pos_tuple in obj if pos_tuple[1] in tags])

        return num_tags if num_tags > 0 else 0

    @staticmethod
    def is_retweet(obj) -> int:

        if not isinstance(obj, str):
            return 0

        return 1 if obj[:3] == 'RT ' else 0

    @staticmethod
    def count_words_intersection(list1: list, list2: list) -> int:
        if not list1 or not list2:
            return 0

        set1_lower = {x.lower() for x in list1}
        set2_lower = {y.lower() for y in list2}

        if not set1_lower or not set2_lower:
            return 0

        # Count number of words in set intersection
        num_common_words = len(set1_lower & set2_lower)

        return num_common_words / len(set1_lower | set2_lower) if num_common_words > 0 else 0
