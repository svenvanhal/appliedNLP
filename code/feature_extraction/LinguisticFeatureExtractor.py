import re


class CharacterCounter:
    """
    ASSUMPTIONS:
        - Special characters do count for the `character length' but not as a word
        - The `diff' methods return the absolute value

    OBSERVATIONS:
        - postTitle does not exist, just postText --> "[<text of the post with links removed>]". Analysis showed that the postText contains at most 1 element (so Y U an array?)

    FEATURE IDEAS:
        - Post Title same as Article Title (0,1)
            Maybe preprocess retweets e.g. [RT @fionamatthias: 10 ways the expat life Is like a continual espresso buzz via @WSJ] -> 10 Ways the Expat Life Is Like a Continual Espresso Buzz - Expat
    """

    def chars(self, string):
        """Determine the number of characters in a string."""

        # Strip spaces
        processed = string.strip().replace(" ", "")

        return len(processed)

    def diff(self, string, other_string):
        """Calculate the (absolute) difference in number of characters between two strings."""

        return abs(self.chars(string) - self.chars(other_string))


class WordCounter:
    """
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

    # Regex pattern for all relevant punctuation
    puncpat = re.compile(r"[,;@#?!&$\"\-\-]+ *")

    def words(self, sentence):
        """Return the word count of a string."""

        # Replace punctuation with space
        sentence = self._clean(sentence)

        return len(sentence.split())

    def diff(self, sentence, other_sentence):
        """Calculate the (absolute) difference in number of words between two sentences."""

        return abs(self.words(sentence) - self.words(other_sentence))

    def _clean(self, sentence):
        """Replace punctuation with a space."""

        return self.puncpat.sub(" ", sentence)
