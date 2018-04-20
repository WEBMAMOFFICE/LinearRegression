import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

categories = {1: "True", 2: "False", 3: "Neutral", 4: "Hint",
              5: "Philosophy", 6: "Question", 7: "Hypothesis"}
X_tasks1 = ["We are", "We are not", "We have not", "We have",
            "We are sometimes", "We have ?", "We think"]
y_tasks1 = ["We are not", "We think"]
tfidfve = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict', dtype='float64',
                          encoding='utf-8', input='content', lowercase=True, max_df=1.0,
                          max_features=None, min_df=1, ngram_range=(1, 1), norm='l2',
                          preprocessor=None, smooth_idf=True, stop_words=None, strip_accents=None,
                          sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b',
                          tokenizer=None, use_idf=True, vocabulary=None)
lr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
tfidfve_fit1 = tfidfve.fit_transform(X_tasks1, list(categories.keys()))
lrf1 = lr.fit(tfidfve_fit1, list(categories.keys()))
print("coef_ value = %s" % (lrf1.coef_,))
print("intercept_ value = %s" % (lrf1.intercept_,))
#print("Coeficient of Determination R² = %s" % (round(lrf1.score(X_tasks1, categories), 4),))
tfidfve_new1 = tfidfve.transform(y_tasks1)
lrp1 = lr.predict(tfidfve_new1)
print("\n--- Prediction # 1")
for cat, pr in zip(y_tasks1, lrp1):
    num = pr
    if 0 > pr < 2:
        pr = 1
    elif 2 >= pr < 3:
        pr = 2
    elif 3 >= pr < 4:
        pr = 3
    elif 4 >= pr < 5:
        pr = 4
    elif 5 >= pr < 6:
        pr = 5
    else:
        pr = 6
    print("%10s%s%-10s (%s)" % (cat, "--->".center(10), categories.get(pr), num))

# Plot outputs
