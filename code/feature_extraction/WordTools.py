import re

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

    def preprocess(self, sentence):
        # TODO: speed up this method

        # Remove hashtag and at symbols
        # TODO: check if this if too harsh, maybe switch out for a regex (but is hard!!)
        sentence = sentence.replace("#", "")
        sentence = sentence.replace("@", "")

        # Convert unrecognized unicode apostrophes back to regular ones
        sentence = sentence.replace("‘", "'")
        sentence = sentence.replace("’", "'")

        # Convert encoded &amp; sign back
        sentence = sentence.replace("&amp;", "&")

        return sentence

    def process(self, sentence, remove_stopwords=False, digit_as_word=True):

        if not isinstance(sentence, str):
            raise ValueError("Word features can only be extracted from a single string.")

        # Convert string to tokens
        tokens = word_tokenize(self.preprocess(sentence))

        # Optional: remove stop words
        if remove_stopwords:
            tokens = self.__clear_stopwords(tokens)

        # Get PoS features (and map to WordNet tags)
        # See: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        pos = pos_tag(tokens)

        # Remove punctuation (PoS tag '.')
        pos_nopunc = self.__filter_tags(pos, {'.', ':', ',', "''", '$', "``", "(", ")"})

        # Optional: remove digits (PoS tag 'CD' - Cardinal Digit)
        if not digit_as_word:
            pos_nopunc = self.__filter_tags(pos_nopunc, {'CD'})

        # Map PoS tags to WordNet tags
        wn_pos = list(map(self.__pos_tags_to_wordnet, pos_nopunc))

        # Create all words return list
        all_words = [item[0] for item in wn_pos]

        # Lemmatize
        lemmas = [self.lem.lemmatize(word, tag) for word, tag in wn_pos]

        # Check lemmas in WordNet
        formal_words = [lemma for lemma in lemmas if wn.synsets(lemma)]

        return [all_words, formal_words]

    def __pos_tags_to_wordnet(self, word_tag):
        """
        Converts default NLTK PoS tags to WordNet-compatible tags.
        Inspired by: https://stackoverflow.com/a/35465203
        """

        try:
            tag = self.morphy_tag[word_tag[1][:2]]
        except:
            tag = wn.NOUN

        return word_tag[0], tag

    def __clear_stopwords(self, words):
        """Removes stop words from the provided word list."""

        return [word for word in words if word not in self.stopwords]

    def __filter_tags(self, list_of_tuples, tags: set) -> list:
        """Removes words from list of word/tag tuples if tag matches function argument."""

        return [word_tag for word_tag in list_of_tuples if word_tag[1] not in tags]

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
