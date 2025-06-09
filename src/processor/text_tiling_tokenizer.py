from nltk.tokenize import TextTilingTokenizer, sent_tokenize

class CustomTextTilingTokenizer(TextTilingTokenizer):
    """
    A wrapper around NLTK's TextTilingTokenizer
    Current functionality:
    
    keeps:
        • self.depth_scores   – valley depths (len = #gaps)
        • self.boundaries     – 0/1 flag per gap (len = #gaps)
 
    The regular return value (a list of tiles) is unchanged.
    """
    def _depth_scores(self, scores):
        self.depth_scores = super()._depth_scores(scores)
        return self.depth_scores

    def _identify_boundaries(self, depth_scores):
        self.boundaries = super()._identify_boundaries(depth_scores)
        return self.boundaries