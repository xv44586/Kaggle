"""
"""
__author__ = 'xumingming'


def count_group_by(df, features, new_feature_name=None):
    new_feature_name = new_feature_name if new_feature_name else reduce(lambda a,b: a + '_' + b, features)
    gp_result = df[features].groupby(by=features[:-1])[features[-1]].count().reset_index().rename(
        index=str,
        columns={features[-1]: new_feature_name}
    )
    return gp_result, new_feature_name
