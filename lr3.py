from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

categories = {1: "True", 2: "False", 3: "Neutral", 4: "Hint",
              5: "Philosophy", 6: "Question", 7: "Hypothesis"}
X_tasks1 = list(categories.values())
y_tasks1 = ["We are students, it is True !", "We think this is Philosophy.", "Forward.",
            "It is absolutlely False.", "Good Question.", "This Hypothesis is False.",
            "Interested Hypothesis."]
y_tasks2 = ["Students, it is True or False!", "Philosophy have a lot of Questions.",
            "This Hint may be True or False.", "False Hypothesis is not a Question.",
            "Your Question was good, but False.", "This Hint is Neutral.",
            "False Philosophy it is a True."]
tfidfve = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict', dtype='float64',
                          encoding='utf-8', input='content', lowercase=True, max_df=1.0,
                          max_features=None, min_df=1, ngram_range=(1, 1), norm='l2',
                          preprocessor=None, smooth_idf=True, stop_words=None, strip_accents=None,
                          sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b',
                          tokenizer=None, use_idf=True, vocabulary=None)
lr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
tfidfve_fit1 = tfidfve.fit_transform(X_tasks1, list(categories.keys()))
lr.fit(tfidfve_fit1, list(categories.keys()))
print("coef_ value = %s" % (lr.coef_,))
print("intercept_ value = %s" % (lr.intercept_,))
print("\n--- Prediction:\n")


def predict_func(data=None, task=None, categories=None):
    for cat, pr in zip(data, task):
        num = float(pr)
        if 0 >= int(pr) < 2:
            pr = 1
        elif 2 >= int(pr) < 3:
            pr = 2
        elif 3 >= int(pr) < 4:
            pr = 3
        elif 4 >= int(pr) < 5:
            pr = 4
        elif 5 >= int(pr) < 6:
            pr = 5
        elif 6 >= int(pr) < 7:
            pr = 6
        else:
            pr = 7
        print("%35s%s%-20s (%s)" % (cat, "--->".center(10), categories.get(pr), num))


tfidfve_new1 = tfidfve.transform(y_tasks1)
lrp1 = lr.predict(tfidfve_new1)
predict_func(data=y_tasks1, task=lrp1, categories=categories)
tfidfve_new2 = tfidfve.transform(y_tasks2)
lrp2 = lr.predict(tfidfve_new2)
predict_func(data=y_tasks2, task=lrp2, categories=categories)
