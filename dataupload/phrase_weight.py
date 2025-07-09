class PhraseWeightScorer:
    def __init__(self, df_thresholds=(1, 2, 3), pmi_threshold=3.0, boost_factors=(0.2, 0.1, 0.0, -0.5)):
        self.df_thresholds = df_thresholds  # (高档, 中档, 低档上限)
        self.pmi_threshold = pmi_threshold
        self.boost_factors = boost_factors  # (高档, 中档, 低档, 噪声)
    def get_boost(self, df, pmi):
        if pmi is not None:
            if pmi < self.pmi_threshold:
                return self.boost_factors[3]  # 噪声
        if df == 1:
            return self.boost_factors[0]
        if 2 <= df <= self.df_thresholds[1]:
            return self.boost_factors[1]
        if self.df_thresholds[1] < df <= self.df_thresholds[2]:
            return self.boost_factors[2]
        return self.boost_factors[3]
    def get_weight(self, idf, df, pmi):
        boost = self.get_boost(df, pmi)
        return idf * (1 + boost)

class AutoThresholdPhraseWeightScorer(PhraseWeightScorer):
    """
    支持自动阈值推荐的分档scorer，根据DF分布和分位点自动计算阈值。
    """
    def __init__(self, df_values, mid_quantile=10, low_quantile=33, pmi_threshold=3.0, boost_factors=(0.2, 0.1, 0.0, -0.5)):
        import numpy as np
        df_values = list(df_values)
        if not df_values:
            high = mid = low = 1
        else:
            high = 1
            mid = int(np.percentile(df_values, mid_quantile))
            low = int(np.percentile(df_values, low_quantile))
        super().__init__(df_thresholds=(high, mid, low), pmi_threshold=pmi_threshold, boost_factors=boost_factors)
    @staticmethod
    def from_df_values(df_dict, mid_quantile=10, low_quantile=33, pmi_threshold=3.0, boost_factors=(0.2,0.1,0.0,-0.5)):
        """
        通过DF分布字典自动生成AutoThresholdPhraseWeightScorer
        """
        return AutoThresholdPhraseWeightScorer(list(df_dict.values()), mid_quantile, low_quantile, pmi_threshold, boost_factors) 