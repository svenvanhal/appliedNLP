from collections import namedtuple

from nltk import download, word_tokenize, pos_tag, WordNetLemmatizer, ngrams
from nltk.data import find
from nltk.corpus import wordnet as wn, stopwords as sw

WTReturn = namedtuple('WTReturn', ['words', 'formal_words', 'stopwords', 'pos', 'bigrams', 'trigrams'])
rng_WTReturn = range(0, len(WTReturn._fields))


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
        self.stopwords = sw.words('english')

    def preprocess(self, sentence):

        # Remove hashtag and at symbols
        # TODO: check if this if too harsh, maybe switch out for a regex (but is hard!!)
        sentence = sentence.replace("#", "").replace("@", "")

        # Convert unrecognized unicode apostrophes back to regular ones
        sentence = sentence.replace("‘", "'").replace("’", "'")
        sentence = sentence.replace("“", '"').replace("”", '"')

        # Convert encoded &amp; sign back
        sentence = sentence.replace("&amp;", "&")

        return sentence

    def process(self, sentence, remove_digits=False, remove_stopwords=False):
        """
        Preprocess string, tokenize, get PoS tags, lookup lemmatized words in WordNet and return:
            - All words (tokens) in the sentence
            - All formal words in the sentence (lemmas found in WordNet)
            - Part-of-Speech tags.

        Optionally filters stopwords and/or cardinal digits.
        """

        if not isinstance(sentence, str):
            return self.process_list(sentence, remove_digits, remove_stopwords)

        # Convert string to tokens
        tokens = word_tokenize(self.preprocess(sentence))

        # Get PoS tags (and map to WordNet tags)
        # See: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        pos_raw = pos_tag(tokens)

        # Remove punctuation
        pos = self.__filter_tags(pos_raw, {'.', ':', ',', "''", '$', "``", "(", ")"})

        # Optionally remove digits (PoS tag 'CD' - Cardinal Digit)
        if not remove_digits:
            pos = self.__filter_tags(pos, {'CD'})

        # Split words and stop words, optionally remove from original word list
        pos, stopwords = self.__split_stopwords(pos, remove_stopwords)

        # Get just the words from the PoS word/tag tuples
        all_words = [item[0] for item in pos]

        # Generate 2- and 3-grams
        bigrams, trigrams = self.__get_ngrams(all_words, 2, 3)

        # Map PoS tags to WordNet tags, lemmatize and find lemmas in WordNet
        wn_pos = list(map(self.__pos_tags_to_wordnet, pos))
        lemmas = [self.lem.lemmatize(word, tag) for word, tag in wn_pos]
        formal_words = [lemma for lemma in lemmas if wn.synsets(lemma)]

        return WTReturn(all_words, formal_words, stopwords, pos, bigrams, trigrams)

    def process_list(self, sentence_list, remove_digits=False, remove_stopwords=False):

        results = map(lambda x: self.process(x, remove_digits, remove_stopwords), sentence_list)
        merged = tuple([i[x] for i in results] for x in rng_WTReturn)
        return WTReturn(*merged)

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

    def __split_stopwords(self, words, remove=False):
        """
        Splits stop words from the provided word list.
        Optionally removes the words from the original list.
        """

        stopwords = []
        filter_words = []

        for pos_tuple in words:

            if pos_tuple[0] in self.stopwords:
                stopwords.append(pos_tuple)
            elif not remove:
                filter_words.append(pos_tuple)

        if remove:
            return filter_words, stopwords
        else:
            return words, stopwords

    def __filter_tags(self, list_of_tuples, tags: set) -> list:
        """Removes words from list of word/tag tuples if tag matches function argument."""

        return [word_tag for word_tag in list_of_tuples if word_tag[1] not in tags]

    def __get_ngrams(self, words, n1, n2):

        n1gram = list(ngrams(words, n1))
        n2gram = list(ngrams(words, n2))

        return n1gram, n2gram

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
