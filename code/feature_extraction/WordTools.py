import re

from itertools import chain

from nltk import download, word_tokenize, pos_tag, WordNetLemmatizer
from nltk.data import find
from nltk.corpus import wordnet as wn, stopwords


class WordTools:
    """
    Processes sentences to extract and count words.
    """

    morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                  'VB': wn.VERB, 'RB': wn.ADV}

    def __init__(self):

        # Download required NLTK libraries
        self.__nltk_init()

        self.lem = WordNetLemmatizer()
        self.stopwords = stopwords.words('english')

    def get_word_features(self, sentence):

        if not isinstance(sentence, str):
            raise ValueError("Word features can only be extracted from a single string.")

        # Convert string to tokens
        tokens = word_tokenize(sentence)

        # Get PoS features (and map to WordNet tags)
        pos = pos_tag(tokens)
        wn_pos = list(map(self.__pos_tags_to_wordnet, pos))

        # Lemmatize and remove stop words
        lemmas = [self.lem.lemmatize(word, tag) for word, tag in wn_pos if word not in self.stopwords]

        # Check lemmas in WordNet
        formal = [lemma for lemma in lemmas if wn.synsets(lemma)]

        return [tokens, pos, wn_pos, lemmas, formal]

    def __pos_tags_to_wordnet(self, tuple):
        """Inspired by: https://stackoverflow.com/a/35465203"""

        try:
            tag = self.morphy_tag[tuple[1][:2]]
        except:
            tag = wn.NOUN

        return tuple[0], tag

    def __nltk_init(self):
        """Download and install NLTK resources if not found on the system."""

        try:
            find('corpora/wordnet.zip')
        except LookupError:
            download('wordnet')

        try:
            find('tokenizers/punkt/english.pickle')
        except LookupError:
            download('punkt')

        try:
            find('taggers/averaged_perceptron_tagger.zip')
        except LookupError:
            download('averaged_perceptron_tagger')

        try:
            find('corpora/stopwords.zip')
        except LookupError:
            download('stopwords')

    # ----

    # Regex pattern for all relevant punctuation
    dashpat = re.compile(r"-")
    doubledashpat = re.compile(r"--")  # TODO: 2 or more dashes
    puncpat = re.compile(r"[,;@#?!&$\"]+ *")

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

    def count_words(self, obj):
        """
        Return the word count of a string.
        Wrapper around len(words) to accommodate lists of words (for which the count is averaged)
        Returns -1 for empty lists or non-existent strings. 0 for empty strings.
        """

        # Catch empty sentences (also catches empty list)
        if not obj and obj != "":
            return -1

        # Get all (lowercased) words in string or list
        words = self.get_words(obj)

        # Average the number of words in a list
        if not isinstance(obj, str):
            return len(words) / len(obj)

        # Return the number of words in the sentence
        return len(words)

    def formal_words(self, distinct_words) -> set:
        """
        Returns a set of all formal words, using WordNet via NLTK.
        N.B.: Is case-sensitive!
        """

        # Get the set of all words which exist in the WordNet corpus
        return set(word for word in distinct_words if wn.synsets(word))
