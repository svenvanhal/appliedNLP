from collections import OrderedDict
from itertools import combinations

import pandas as pd

from .WordTools import WordTools
from .Util import Util
from .ImageHelper import ImageHelper


class FeatureExtractor:
    required_cols = ['postText', 'targetKeywords', 'targetDescription', 'targetTitle', 'targetParagraphs']

    df = None

    def __init__(self, data_path, tesseract_path):
        self.wordtools = WordTools()
        self.imagehelper = ImageHelper(data_path, tesseract_path)

    def set_df(self, df: pd.DataFrame) -> None:
        """
        Sets dataframe to extract features from.
        """

        # Check required columns
        if not set(self.required_cols).issubset(df.columns):
            raise ValueError("DataFrame does not contain all required columns ('%s')" % "', '".join(self.required_cols))

        # Copy df to encapsulate changes in this class
        self.df = df.copy()

    def extract_features(self, char_based=True, word_based=True, pos_based=True, sim_based=True, debug=True):
        """
        Extracts the relevant features from a Pandas dataframe.
        """

        if self.df is None:
            raise ValueError(
                "No dataframe defined. Please call " + '\033[1m' + "FeatureExtractor.set_df()" + '\033[0m' + " first.")

        # Get targets
        labels = self.__get_targets(self.df['truthClass'])

        # TODO: optionally convert to dask partitions and parellelize
        # ddata = dd.from_pandas(self.df, npartitions=8)
        # features = ddata.map_partitions(
        #     lambda df: df.apply(lambda x: self.__get_features(x, char_based, word_based, pos_based, sim_based, debug),
        #                         axis=1)).compute(scheduler='threads')

        # Get features
        features = self.df.apply(lambda x: self.__get_features(x, char_based, word_based, pos_based, sim_based, debug),
                                 axis=1)

        return labels, features

    def __get_targets(self, truth_classes: pd.Series) -> pd.Series:
        """
        Maps categorical truth classes to integer targets.
        """

        labels, _ = pd.factorize(truth_classes, sort=False)
        return labels

    def dict2feature(self, features, name: str, data: dict) -> None:
        """Append feature name to dict key and append to features."""

        for k, v in data.items():
            features["{}_{}".format(name, k)] = v

    def combi_dict2feature(self, features, name: str, data: dict, func) -> None:
        """More efficiently calculate features for all combinations."""

        for var1, var2 in combinations(data, 2):
            features["{}_{}_{}".format(name, var1, var2)] = func(data[var1], data[var2])

    def __get_features(self, row, char_based=True, word_based=True, pos_based=True, sim_based=True, debug=True):
        """
        Extracts features from dataset row.

        TODO: check if it makes sense to calculate the average keyword length as opposed to the total word length: says so in the paper, but seems strange
        """

        features = pd.Series()

        # ------

        # Get relevant fields
        post_title = row['postText'][0]  # Assumption: postText always has one item
        article_title = row['targetTitle']
        post_media = row['postMedia']
        article_kw = row['targetKeywords']
        article_desc = row['targetDescription']
        article_par = row['targetParagraphs']

        # Prep
        proc_post_title = self.wordtools.process(post_title)
        proc_article_title = self.wordtools.process(article_title)
        post_image = self.imagehelper.get_text(post_media)
        proc_post_image = self.wordtools.process(post_image)
        # proc_article_kw = self.wordtools.process(article_kw)
        # proc_article_desc = self.wordtools.process(article_desc)
        # proc_article_par = self.wordtools.process_list(article_par)

        if debug:
            features['proc_post_title'] = proc_post_title
            features['proc_article_title'] = proc_article_title

            return features

        if char_based:
            # Calculate num characters
            num_chars = OrderedDict()
            num_chars['post_title'] = Util.count_chars(post_title)
            num_chars['article_title'] = Util.count_chars(article_title)
            num_chars['post_image'] = Util.count_chars(post_image)
            num_chars['article_kw'] = Util.count_chars(article_kw)
            num_chars['article_desc'] = Util.count_chars(article_desc)
            num_chars['article_par'] = Util.count_chars(article_par)

            # Calculate num question marks
            num_qmarks = OrderedDict()
            num_qmarks['post_title'] = Util.count_specific_char(post_title, '?')
            num_qmarks['article_title'] = Util.count_specific_char(article_title, '?')
            num_qmarks['post_image'] = Util.count_specific_char(post_image, '?')
            num_qmarks['article_keywords'] = Util.count_specific_char(article_kw, '?')
            num_qmarks['article_desc'] = Util.count_specific_char(article_desc, '?')
            num_qmarks['article_par'] = Util.count_specific_char(article_par, '?')

            # Generate features
            self.dict2feature(features, 'numChars', num_chars)
            self.dict2feature(features, 'numQuestionMarks', num_qmarks)
            self.combi_dict2feature(features, 'ratioChars', num_chars, Util.ratio)
            self.combi_dict2feature(features, 'diffChars', num_chars, Util.diff)

        if word_based:
            # Calculate num words
            num_words = OrderedDict()
            num_words['post_title'] = Util.count_words(proc_post_title.words)
            num_words['article_title'] = Util.count_words(proc_article_title.words)
            num_words['post_image'] = Util.count_words(proc_post_image.words)
            # num_words['article_kw'] = Util.count_words(proc_article_kw.words)
            # num_words['article_desc'] = Util.count_words(proc_article_desc.words)
            # num_words['article_par'] = Util.count_words(proc_article_par.words)

            # Calculate num uppercase words
            num_uppercase = OrderedDict()
            num_uppercase['post_title'] = Util.count_words_uppercase(proc_post_title.words)
            num_uppercase['article_title'] = Util.count_words_uppercase(proc_article_title.words)
            num_uppercase['post_image'] = Util.count_words_uppercase(proc_post_image.words)

            # Calculate num titlecased words (first letter capitalized)
            num_titlecase = OrderedDict()
            num_titlecase['post_title'] = Util.count_words_titlecase(proc_post_title.words)
            num_titlecase['article_title'] = Util.count_words_titlecase(proc_article_title.words)
            num_titlecase['post_image'] = Util.count_words_titlecase(proc_post_image.words)

            # Calculate num formal words
            num_formal_words = OrderedDict()
            num_formal_words['post_title'] = Util.count_words(proc_post_title.formal_words)
            num_formal_words['article_title'] = Util.count_words(proc_article_title.formal_words)
            num_formal_words['post_image'] = Util.count_words(proc_post_image.formal_words)
            # num_formal_words['article_kw'] = Util.count_words(proc_article_kw.formal_words)
            # num_formal_words['article_desc'] = Util.count_words(proc_article_desc.formal_words)
            # num_formal_words['article_par'] = Util.count_words(proc_article_par.formal_words)

            # Calculate num stop words
            num_stopwords = OrderedDict()
            num_stopwords['post_title'] = Util.count_words(proc_post_title.stopwords)
            num_stopwords['article_title'] = Util.count_words(proc_article_title.stopwords)
            num_stopwords['post_image'] = Util.count_words(proc_post_image.stopwords)
            # num_stopwords['article_kw'] = Util.count_words(proc_article_kw.stopwords)
            # num_stopwords['article_desc'] = Util.count_words(proc_article_desc.stopwords)
            # num_stopwords['article_par'] = Util.count_words(proc_article_par.stopwords)

            # Generate features
            self.dict2feature(features, 'numWords', num_words)
            self.dict2feature(features, 'numWordsUppercase', num_uppercase)
            self.dict2feature(features, 'numWordsTitlecase', num_titlecase)
            self.dict2feature(features, 'numFormalWords', num_formal_words)
            self.dict2feature(features, 'numStopWords', num_stopwords)
            self.combi_dict2feature(features, 'ratioWords', num_words, Util.ratio)
            self.combi_dict2feature(features, 'ratioFormalWords', num_formal_words, Util.ratio)
            self.combi_dict2feature(features, 'ratioStopWords', num_stopwords, Util.ratio)
            self.combi_dict2feature(features, 'diffWords', num_words, Util.diff)
            self.combi_dict2feature(features, 'diffFormalWords', num_formal_words, Util.diff)
            self.combi_dict2feature(features, 'diffStopWords', num_stopwords, Util.diff)

        if pos_based:

            tags = [{'NNP'}, {'DT'}, {'PRP'}]

            for tag_set in tags:
                # Count tags
                num_pos_tags = OrderedDict()
                num_pos_tags['post_title'] = Util.count_tags(proc_post_title.pos, tag_set)
                num_pos_tags['article_title'] = Util.count_tags(proc_article_title.pos, tag_set)

                # Generate features
                self.dict2feature(features, 'numTags' + repr(tag_set), num_pos_tags)
                self.combi_dict2feature(features, 'ratioTags_' + repr(tag_set), num_pos_tags, Util.ratio_raw)
                self.combi_dict2feature(features, 'diffTags_' + repr(tag_set), num_pos_tags, Util.diff_raw)

        return features
