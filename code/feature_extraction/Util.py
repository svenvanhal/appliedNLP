class Util:
    """
    CharCounter:

        ASSUMPTIONS:
            - Special characters do count for the `character length' but not as a word
            - The `diff' methods return the absolute value

        OBSERVATIONS:
            - postTitle does not exist, just postText --> "[<text of the post with links removed>]". Analysis showed that the postText contains at most 1 element (so Y U an array?)

        FEATURE IDEAS:
            - Post Title same as Article Title (0,1)
                Maybe preprocess retweets e.g. [RT @fionamatthias: 10 ways the expat life Is like a continual espresso buzz via @WSJ] -> 10 Ways the Expat Life Is Like a Continual Espresso Buzz - Expat
            - Match known clickbait indicators
                https://www.quora.com/What-are-the-most-commonly-used-words-in-clickbait-titles-Those-news-headlines-that-go-like-You-wont-believe-___-or-This-___-will-restore-your-faith-in-humanity-X-easy-tricks-to-___-They-hint-at-the-article-and-tempt-you-to-click
                https://www.cnet.com/news/facebook-nixes-click-bait-headlines-in-users-news-feeds/

    WordCounter:
        WORD COUNT -- Harder Than It Sounds
            Original approach: len(sentence.split()), but this also counts punctuation (e.g. '--' or '&') as words
            Second approach: regex  re.compile(r'\S+')  (failed to recognize decimal numbers)
            Third approach: cleaning text and split on space, problem: string.punctuation too broad
              --> (sentence.translate(str.maketrans(' ', ' ', string.punctuation)))

        So, problems:
            word--word     (are two words)
            U.S.           (is one word)
            2.5            (is one 'word')
            missing'space' (are two words)
    """

    @staticmethod
    def ratio(left, right):

        # Catch edge cases
        # TODO: check if `<= 0' check makes sense
        if not left or left <= 0 or not right or right <= 0:
            return -1

        return abs(left / right)

    @staticmethod
    def diff(left, right):

        # Catch edge cases
        # TODO: check if `<= 0' check makes sense
        if not left or left <= 0 or not right or right <= 0:
            return -1

        return abs(left - right)

    @staticmethod
    def count_chars(obj):
        """Determine the number of characters in a string."""

        # Catch empty post titles
        if obj is None:
            return -1

        # Catch non-strings (probably list / pd.Series)
        if not isinstance(obj, str):

            # Catch empty lists
            if len(obj) == 0:
                return -1

            # Average the length of items in a list
            return sum(map(Util.count_chars, obj)) / len(obj)

        # Strip spaces
        processed = obj.strip().replace(" ", "")

        # Return string length
        return len(processed)
