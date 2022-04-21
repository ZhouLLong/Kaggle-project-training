# Vectorizer:
from sklearn.feature_extraction.text import TfidfVectorizer
def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

vectorizer=TfidfVectorizer(tokenizer=word_divide_char)
## TfidfVectorizer(tokenizer=<function word_divide_char at 0x7f9cacf3fdc0>)

X=vectorizer.fit_transform(X)

# models:
from sklearn.linear_model import LogisticRegression

## default:
LogisticRegression(penalty='l2',multi_class='ovr')

## for multi:
LogisticRegression(multi_class='multinomial', random_state=0,
                   solver='newton-cg')


import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(eval_metric='mlogloss')


## default:
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              eval_metric='mlogloss', gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=100, n_jobs=4,
              num_parallel_tree=1, objective='multi:softprob', predictor='auto',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)
