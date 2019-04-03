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
        image_text = self.imagehelper.get_text(post_media)

        # @formatter:off
        w_post_title,    fw_post_title,    sw_post_title,    pos_post_title    = self.wordtools.process(post_title)
        w_post_image,    fw_post_image,    sw_post_image,    pos_post_image    = self.wordtools.process(image_text)
        w_article_kw,    fw_article_kw,    sw_article_kw,    pos_article_kw    = self.wordtools.process(article_kw)
        w_article_descr, fw_article_descr, sw_article_descr, pos_article_descr = self.wordtools.process(article_descr)
        w_article_title, fw_article_title, sw_article_title, pos_article_title = self.wordtools.process(article_title)
        # @formatter:on

        # Calculate num characters
        nc_post_title = Util.count_chars(post_title)
        nc_post_image = Util.count_chars(image_text)
        nc_article_keywords = Util.count_chars(article_kw)
        nc_article_desc = Util.count_chars(article_descr)
        nc_article_title = Util.count_chars(article_title)
        nc_article_paragraphs = Util.count_chars(article_paragraphs)

        # Calculate num words
        nw_post_title = len(w_post_title)
        nw_post_image = len(w_post_image)
        nw_article_keywords = len(w_article_kw)
        nw_article_description = len(w_article_descr)
        nw_article_title = len(w_article_title)
        # Note: num words article paragraphs currently not supported!

        # Calculate num formal words
        nfw_post_title = len(fw_post_title)

        # Calculate num stop words
        nsw_post_title = len(sw_post_title)

        # Calculate PoS features
        pos_nn_post_title = Util.count_tags(pos_post_title, {'NN'})

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
        features['numFormalWords_PostTitle'] = nfw_post_title

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

        # ------

        # Number of stop words in Post Title
        features['numStopWords_PostTitle'] = nsw_post_title

        # Number of "Noun, singular or mass" in Post Title
        features['numTags_NN_PostTitle'] = pos_nn_post_title

        return features
