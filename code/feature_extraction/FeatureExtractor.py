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

    def extract_features(self):
        """
        Extracts the relevant features from a Pandas dataframe.
        """

        if self.df is None:
            raise ValueError(
                "No dataframe defined. Please call " + '\033[1m' + "FeatureExtractor.set_df()" + '\033[0m' + " first.")

        # Get targets
        labels = self.__get_targets(self.df['truthClass'])

        # Get features
        features = self.df.apply(self.__get_features, axis=1)

        return labels, features

    def __get_targets(self, truth_classes: pd.Series) -> pd.Series:
        """
        Maps categorical truth classes to integer targets.
        """

        labels, _ = pd.factorize(truth_classes, sort=False)
        return labels

    def __get_features(self, row: pd.Series) -> pd.Series:
        """
        Extracts features from dataset row.
        N.B.: new approach does not average the word length for arrays of text (e.g. article paragraphs)!
        """

        features = pd.Series()

        # ------

        # Get relevant fields
        post_title = row['postText'][0]  # Assumption: postText always has one item
        post_media = row['postMedia']
        article_kw = row['targetKeywords']
        article_descr = row['targetDescription']
        article_title = row['targetTitle']
        article_paragraphs = row['targetParagraphs']

        # Prep
        post_image = self.imagehelper.get_text(post_media)

        # @formatter:off
        w_post_title,    fw_post_title,    sw_post_title,    pos_post_title    = self.wordtools.process(post_title)
        w_post_image,    fw_post_image,    sw_post_image,    pos_post_image    = self.wordtools.process(post_image)
        w_article_kw,    fw_article_kw,    sw_article_kw,    pos_article_kw    = self.wordtools.process(article_kw)
        w_article_descr, fw_article_descr, sw_article_descr, pos_article_descr = self.wordtools.process(article_descr)
        w_article_title, fw_article_title, sw_article_title, pos_article_title = self.wordtools.process(article_title)

        # Calculate num characters
        nc_post_title         = Util.count_chars(post_title)
        nc_post_image         = Util.count_chars(post_image)
        nc_article_kw   = Util.count_chars(article_kw)
        nc_article_descr      = Util.count_chars(article_descr)
        nc_article_title      = Util.count_chars(article_title)
        nc_article_paragraphs = Util.count_chars(article_paragraphs)

        # Calculate num words
        nw_post_title    = len(w_post_title)
        nw_post_image    = len(w_post_image)
        nw_article_kw    = len(w_article_kw)
        nw_article_descr = len(w_article_descr)
        nw_article_title = len(w_article_title)
        # Note: num words article paragraphs currently not supported!

        # Calculate num formal words
        nfw_post_title    = len(fw_post_title)
        nfw_post_image    = len(fw_post_image)
        nfw_article_kw    = len(fw_article_kw)
        nfw_article_descr = len(fw_article_descr)
        nfw_article_title = len(fw_article_title)

        # Calculate num stop words
        nsw_post_title    = len(sw_post_title)
        nsw_post_image    = len(sw_post_image)
        nsw_article_kw    = len(sw_article_kw)
        nsw_article_descr = len(sw_article_descr)
        nsw_article_title = len(sw_article_title)

        # Calculate PoS features
        pos_nn_post_title = Util.count_tags(pos_post_title, {'NN'})

        # ------
        # TODO: remove amount of "duplicate" code here

        # Number of characters
        features['numChars_post_title']         = nc_post_title
        features['numChars_post_image']         = nc_post_image
        features['numChars_article_keywords']   = nc_article_kw
        features['numChars_article_desc']       = nc_article_descr
        features['numChars_article_title']      = nc_article_title
        features['numChars_article_paragraphs'] = nc_article_paragraphs

        # Number of words
        features['numWords_post_title']       = nw_post_title
        features['numWords_post_image']       = nw_post_image
        features['numWords_article_keywords'] = nw_article_kw
        features['numWords_article_desc']     = nw_article_descr
        features['numWords_article_title']    = nw_article_title
        # features['numWords_article_paragraphs'] = nw_article_paragraphs

        # Number of formal words
        features['numFormalWords_post_title']    = nfw_post_title
        features['numFormalWords_post_image']    = nfw_post_image
        features['numFormalWords_article_kw']    = nfw_article_kw
        features['numFormalWords_article_descr'] = nfw_article_descr
        features['numFormalWords_article_title'] = nfw_article_title
        # features['numFormalWords_article_paragraphs'] = nfw_article_paragraphs

        # Number of stop words
        features['numStopWords_post_title']    = nsw_post_title
        features['numStopWords_post_image']    = nsw_post_image
        features['numStopWords_article_kw']    = nsw_article_kw
        features['numStopWords_article_descr'] = nsw_article_descr
        features['numStopWords_article_title'] = nsw_article_title
        # features['numFormalWords_article_paragraphs'] = nfw_article_paragraphs

        # Number of question marks
        features['numQuestionMarks_post_title']         = Util.count_specific_char(post_title, '?')
        features['numQuestionMarks_post_image']         = Util.count_specific_char(post_image, '?')
        features['numQuestionMarks_article_keywords']   = Util.count_specific_char(article_kw, '?')
        features['numQuestionMarks_article_desc']       = Util.count_specific_char(article_descr, '?')
        features['numQuestionMarks_article_title']      = Util.count_specific_char(article_title, '?')
        features['numQuestionMarks_article_paragraphs'] = Util.count_specific_char(article_paragraphs, '?')

        # ------

        # Ratio between number of characters
        features['ratioChars_post_title_post_image']          = Util.ratio(nc_post_title, nc_post_image)
        features['ratioChars_post_title_article_keywords']    = Util.ratio(nc_post_title, nc_article_kw)
        features['ratioChars_post_title_article_descr']       = Util.ratio(nc_post_title, nc_article_descr)
        features['ratioChars_post_title_article_title']       = Util.ratio(nc_post_title, nc_article_title)
        features['ratioChars_post_image_article_keywords']    = Util.ratio(nc_post_image, nc_article_kw)
        features['ratioChars_post_image_article_descr']       = Util.ratio(nc_post_image, nc_article_descr)
        features['ratioChars_post_image_article_title']       = Util.ratio(nc_post_image, nc_article_title)
        features['ratioChars_article_keywords_article_descr'] = Util.ratio(nc_article_kw, nc_article_descr)
        features['ratioChars_article_keywords_article_title'] = Util.ratio(nc_article_kw, nc_article_title)
        features['ratioChars_article_descr_article_title']    = Util.ratio(nc_article_descr, nc_article_title)

        # Ratio between number of words
        features['ratioWords_post_title_post_image']          = Util.ratio(nw_post_title, nw_post_image)
        features['ratioWords_post_title_article_keywords']    = Util.ratio(nw_post_title, nw_article_kw)
        features['ratioWords_post_title_article_descr']       = Util.ratio(nw_post_title, nw_article_descr)
        features['ratioWords_post_title_article_title']       = Util.ratio(nw_post_title, nw_article_title)
        features['ratioWords_post_image_article_keywords']    = Util.ratio(nw_post_image, nw_article_kw)
        features['ratioWords_post_image_article_descr']       = Util.ratio(nw_post_image, nw_article_descr)
        features['ratioWords_post_image_article_title']       = Util.ratio(nw_post_image, nw_article_title)
        features['ratioWords_article_keywords_article_descr'] = Util.ratio(nw_article_kw, nw_article_descr)
        features['ratioWords_article_keywords_article_title'] = Util.ratio(nw_article_kw, nw_article_title)
        features['ratioWords_article_descr_article_title']    = Util.ratio(nw_article_descr, nw_article_title)

        # Ratio between number of formal words
        features['ratioFormalWords_post_title_post_image']          = Util.ratio(nfw_post_title, nfw_post_image)
        features['ratioFormalWords_post_title_article_keywords']    = Util.ratio(nfw_post_title, nfw_article_kw)
        features['ratioFormalWords_post_title_article_descr']       = Util.ratio(nfw_post_title, nfw_article_descr)
        features['ratioFormalWords_post_title_article_title']       = Util.ratio(nfw_post_title, nfw_article_title)
        features['ratioFormalWords_post_image_article_keywords']    = Util.ratio(nfw_post_image, nfw_article_kw)
        features['ratioFormalWords_post_image_article_descr']       = Util.ratio(nfw_post_image, nfw_article_descr)
        features['ratioFormalWords_post_image_article_title']       = Util.ratio(nfw_post_image, nfw_article_title)
        features['ratioFormalWords_article_keywords_article_descr'] = Util.ratio(nfw_article_kw, nfw_article_descr)
        features['ratioFormalWords_article_keywords_article_title'] = Util.ratio(nfw_article_kw, nfw_article_title)
        features['ratioFormalWords_article_descr_article_title']    = Util.ratio(nfw_article_descr, nfw_article_title)

        # Ratio between number of stop words
        features['ratioStopWords_post_title_post_image']          = Util.ratio(nsw_post_title, nfw_post_image)
        features['ratioStopWords_post_title_article_keywords']    = Util.ratio(nsw_post_title, nfw_article_kw)
        features['ratioStopWords_post_title_article_descr']       = Util.ratio(nsw_post_title, nfw_article_descr)
        features['ratioStopWords_post_title_article_title']       = Util.ratio(nsw_post_title, nfw_article_title)
        features['ratioStopWords_post_image_article_keywords']    = Util.ratio(nsw_post_image, nfw_article_kw)
        features['ratioStopWords_post_image_article_descr']       = Util.ratio(nsw_post_image, nfw_article_descr)
        features['ratioStopWords_post_image_article_title']       = Util.ratio(nsw_post_image, nfw_article_title)
        features['ratioStopWords_article_keywords_article_descr'] = Util.ratio(nsw_article_kw, nfw_article_descr)
        features['ratioStopWords_article_keywords_article_title'] = Util.ratio(nsw_article_kw, nfw_article_title)
        features['ratioStopWords_article_descr_article_title']    = Util.ratio(nsw_article_descr, nfw_article_title)

        # ------

        # Diff between number of characters
        features['diffChars_post_title_post_image']          = Util.diff(nc_post_title, nc_post_image)
        features['diffChars_post_title_article_keywords']    = Util.diff(nc_post_title, nc_article_kw)
        features['diffChars_post_title_article_descr']       = Util.diff(nc_post_title, nc_article_descr)
        features['diffChars_post_title_article_title']       = Util.diff(nc_post_title, nc_article_title)
        features['diffChars_post_image_article_keywords']    = Util.diff(nc_post_image, nc_article_kw)
        features['diffChars_post_image_article_descr']       = Util.diff(nc_post_image, nc_article_descr)
        features['diffChars_post_image_article_title']       = Util.diff(nc_post_image, nc_article_title)
        features['diffChars_article_keywords_article_descr'] = Util.diff(nc_article_kw, nc_article_descr)
        features['diffChars_article_keywords_article_title'] = Util.diff(nc_article_kw, nc_article_title)
        features['diffChars_article_descr_article_title']    = Util.diff(nc_article_descr, nc_article_title)

        # Diff between number of words
        features['diffWords_post_title_post_image']          = Util.diff(nw_post_title, nw_post_image)
        features['diffWords_post_title_article_keywords']    = Util.diff(nw_post_title, nw_article_kw)
        features['diffWords_post_title_article_descr']       = Util.diff(nw_post_title, nw_article_descr)
        features['diffWords_post_title_article_title']       = Util.diff(nw_post_title, nw_article_title)
        features['diffWords_post_image_article_keywords']    = Util.diff(nw_post_image, nw_article_kw)
        features['diffWords_post_image_article_descr']       = Util.diff(nw_post_image, nw_article_descr)
        features['diffWords_post_image_article_title']       = Util.diff(nw_post_image, nw_article_title)
        features['diffWords_article_keywords_article_descr'] = Util.diff(nw_article_kw, nw_article_descr)
        features['diffWords_article_keywords_article_title'] = Util.diff(nw_article_kw, nw_article_title)
        features['diffWords_article_descr_article_title']    = Util.diff(nw_article_descr, nw_article_title)

        # Diff between number of formal words
        features['diffFormalWords_post_title_post_image']          = Util.diff(nfw_post_title, nfw_post_image)
        features['diffFormalWords_post_title_article_keywords']    = Util.diff(nfw_post_title, nfw_article_kw)
        features['diffFormalWords_post_title_article_descr']       = Util.diff(nfw_post_title, nfw_article_descr)
        features['diffFormalWords_post_title_article_title']       = Util.diff(nfw_post_title, nfw_article_title)
        features['diffFormalWords_post_image_article_keywords']    = Util.diff(nfw_post_image, nfw_article_kw)
        features['diffFormalWords_post_image_article_descr']       = Util.diff(nfw_post_image, nfw_article_descr)
        features['diffFormalWords_post_image_article_title']       = Util.diff(nfw_post_image, nfw_article_title)
        features['diffFormalWords_article_keywords_article_descr'] = Util.diff(nfw_article_kw, nfw_article_descr)
        features['diffFormalWords_article_keywords_article_title'] = Util.diff(nfw_article_kw, nfw_article_title)
        features['diffFormalWords_article_descr_article_title']    = Util.diff(nfw_article_descr, nfw_article_title)

        # Diff between number of stop words
        features['diffStopWords_post_title_post_image']          = Util.diff(nsw_post_title, nfw_post_image)
        features['diffStopWords_post_title_article_keywords']    = Util.diff(nsw_post_title, nfw_article_kw)
        features['diffStopWords_post_title_article_descr']       = Util.diff(nsw_post_title, nfw_article_descr)
        features['diffStopWords_post_title_article_title']       = Util.diff(nsw_post_title, nfw_article_title)
        features['diffStopWords_post_image_article_keywords']    = Util.diff(nsw_post_image, nfw_article_kw)
        features['diffStopWords_post_image_article_descr']       = Util.diff(nsw_post_image, nfw_article_descr)
        features['diffStopWords_post_image_article_title']       = Util.diff(nsw_post_image, nfw_article_title)
        features['diffStopWords_article_keywords_article_descr'] = Util.diff(nsw_article_kw, nfw_article_descr)
        features['diffStopWords_article_keywords_article_title'] = Util.diff(nsw_article_kw, nfw_article_title)
        features['diffStopWords_article_descr_article_title']    = Util.diff(nsw_article_descr, nfw_article_title)
        # @formatter:on

        # Number of "Noun, singular or mass" in Post Title
        # TODO: enable later
        # features['numTags_NN_PostTitle'] = pos_nn_post_title

        return features
