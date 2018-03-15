"""
"""
__author__ = 'xumingming'

import gc


def count_group_by(df, features, new_feature_name=None, with_time=['day', 'hour']):
    new_feature_name = new_feature_name if new_feature_name else reduce(lambda a, b: a + '_' + b, features)

    features.extend(with_time)

    gp_result = df[features].groupby(by=features[:-1])[features[-1]].count().reset_index().rename(
        index=str,
        columns={features[-1]: new_feature_name}
    )

    drop_features = [c for c in list(df.columns) if c not in features]

    if drop_features:
        df.drop(drop_features, axis=1, replaced=True)
        gc.collect()

    return gp_result, new_feature_name
