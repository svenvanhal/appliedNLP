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
        self.df.drop('truthClass', axis=1, inplace=True)

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
        """

        features = pd.Series()

        # ------

        # Get relevant fields
        post_title = row['postText']
        post_media = row['postMedia']
        article_keywords = row['targetKeywords']
        article_description = row['targetDescription']
        article_title = row['targetTitle']
        article_paragraphs = row['targetParagraphs']

        # Prep
        image_text = self.imagehelper.get_text(post_media)

        w_post_title = self.wordtools.get_words(post_title)
        wu_post_title = list(set(w_post_title))
        # TODO: paper says 'set' of formal words, but wouldn't the 'list' of words make more sense?
        wf_post_title = self.wordtools.formal_words(wu_post_title)

        # Calculate num characters
        nc_post_title = Util.count_chars(post_title)
        nc_post_image = Util.count_chars(image_text)
        nc_article_keywords = Util.count_chars(article_keywords)
        nc_article_desc = Util.count_chars(article_description)
        nc_article_title = Util.count_chars(article_title)
        nc_article_paragraphs = Util.count_chars(article_paragraphs)

        # Calculate num words
        nw_post_title = len(w_post_title)
        nw_post_image = self.wordtools.count_words(image_text)
        nw_article_keywords = self.wordtools.count_words(article_keywords)
        nw_article_description = self.wordtools.count_words(article_description)
        nw_article_title = self.wordtools.count_words(article_title)
        # Unused: nw_article_paragraphs = self.wordtools.count_words(article_paragraphs)

        # Calculate num formal words
        nwf_post_title = len(wf_post_title)

        # ------

        # num of characters in post title
        features['numChars_PostTitle'] = nc_post_title

        # num of characters ratio post image text \& post title
        features['ratioChars_PostImagePostTitle'] = Util.ratio(nc_post_image, nc_post_title)

        # diff num of characters post title \& article keywords
        features['diffChars_PostTitleArticleKeywords'] = Util.diff(nc_post_title, nc_article_keywords)

        # diff num of characters post title \& post image text
        features['diffChars_PostTitlePostImage'] = Util.diff(nc_post_title, nc_post_image)

        # num of words ratio post image text \& post title
        features['ratioWords_PostImagePostTitle'] = Util.ratio(nw_post_image, nw_post_title)

        # num of words in post title
        features['numWords_PostTitle'] = nw_post_title

        # num of formal words in post title
        features['numFormalWords_PostTitle'] = nwf_post_title

        # num of words ratio article description \& post title
        features['ratioWords_ArticleDescPostTitle'] = Util.diff(nw_article_description, nw_post_title)

        # num of characters ratio article description \& post title
        features['ratioChars_ArticleDescPostTitle'] = Util.ratio(nc_article_desc, nc_post_title)

        # num of characters ratio article title \& post title
        features['ratioChars_ArticleTitlePostTitle'] = Util.ratio(nc_article_title, nc_post_title)

        # num of words ratio article title \& post title
        features['ratioWords_ArticleTitlePostTitle'] = Util.ratio(nw_article_title, nw_post_title)

        # diff num of words post title \& article keywords
        features['diffWords_PostTitleArticleKeywords'] = Util.diff(nw_post_title, nw_article_keywords)

        # num of question marks in post title
        features['numQuestionmarksPostTitle'] = Util.count_specific_char(post_title, '?')

        # num of characters ratio article paragraphs \& post title
        features['ratioChars_ArticleParagraphsPostTitle'] = Util.ratio(nc_article_paragraphs, nc_post_title)

        # num of characters ratio article paragraphs \& article desc
        features['ratioChars_ArticleParagraphsArticleDesc'] = Util.ratio(nc_article_paragraphs, nc_article_desc)

        return features
