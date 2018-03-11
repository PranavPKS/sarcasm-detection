import sys
import collections
from collections import defaultdict
import re
import itertools
import sqlite3
import warnings

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import numpy as np
import statsmodels.api as sm

warnings.filterwarnings('ignore')

db_path = "ironate.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

labelers_of_interest = [2,4,5,6]

def make_sql_list_str(ls):
    return "(" + ",".join([str(x_i) for x_i in ls]) + ")"


labeler_id_str = make_sql_list_str(labelers_of_interest)


def grab_single_element(result_set, COL=0):
    return [x[COL] for x in result_set]

def get_all_comment_ids():
    return grab_single_element(cursor.execute('''select distinct comment_id from irony_label where labeler_id in %s;''' %labeler_id_str)) 

def get_ironic_comment_ids():
    cursor.execute('''select distinct comment_id from irony_label where forced_decision=0 and label=1 and labeler_id in %s;''' %labeler_id_str)

    ironic_comments = grab_single_element(cursor.fetchall())
    return ironic_comments




def get_labeled_thrice_comments():
    
    cursor.execute('''select comment_id from irony_label group by comment_id having count(distinct labeler_id) >= 3;''')
    thricely_labeled_comment_ids = grab_single_element(cursor.fetchall())
    return thricely_labeled_comment_ids

def grab_comments(comment_id_list, verbose=False):

    comments_list = []
    for comment_id in comment_id_list:
        cursor.execute("select text from irony_commentsegment where comment_id='%s' order by segment_index" % comment_id)
        segments = grab_single_element(cursor.fetchall())
        comment = " ".join(segments)
        if verbose:
            print comment
        comments_list.append(comment.encode('utf-8').strip())
    return comments_list

def get_entries(a_list, indices):
    return [a_list[i] for i in indices]





    
all_comment_ids = get_all_comment_ids()

# pre-context / forced decisions
forced_decisions = grab_single_element(cursor.execute(
            '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' %labeler_id_str)) 

for labeler in labelers_of_interest:
    labeler_forced_decisions = grab_single_element(cursor.execute(
            '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id = %s;''' %labeler))

    all_labeler_decisions = grab_single_element(cursor.execute(
            '''select distinct comment_id from irony_label where forced_decision=0 and labeler_id = %s;''' %labeler))

    p_labeler_forced = float(len(labeler_forced_decisions))/float(len(all_labeler_decisions))
    print "labeler %s: %s" % (labeler, p_labeler_forced)

p_forced = float(len(forced_decisions)) / float(len(all_comment_ids))

# the proportion forced for the ironic comments
ironic_comments = get_ironic_comment_ids()
ironic_ids_str = make_sql_list_str(ironic_comments)
forced_ironic_ids =  grab_single_element(cursor.execute(
            '''select distinct comment_id from irony_label where 
                    forced_decision=1 and comment_id in %s and labeler_id in %s;''' %(ironic_ids_str, labeler_id_str))) 


X,y = [],[]

for c_id in all_comment_ids:
    if c_id in forced_decisions:
        y.append(1.0)
    else:
        y.append(0.0)

    if c_id in ironic_comments:
        X.append([1.0])
    else:
        X.append([0.0])

X = sm.add_constant(X, prepend=True)
logit_mod = sm.Logit(y, X)
logit_res = logit_mod.fit()

print logit_res.summary()
print logit_res

#--------------------------

all_comment_ids = get_labeled_thrice_comments()

ironic_comment_ids = get_ironic_comment_ids()
#ironic_ids_str = make_sql_list_str(ironic_comments)

forced_decision_ids = grab_single_element(cursor.execute(
            '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' %labeler_id_str)) 

comment_texts, y = [], []
for id_ in all_comment_ids:
    comment_texts.append(grab_comments([id_])[0])
    if id_ in ironic_comment_ids:
        y.append(1)
    else:
        y.append(-1)


emoticon_RE_str = '(?::|;|=)(?:-)?(?:\)|\(|D|P)'
question_mark_RE_str = '\?'
exclamation_point_RE_str = '\!'
# any combination of multiple exclamation points and question marks
interrobang_RE_str = '[\?\!]{2,}'

for i, comment in enumerate(comment_texts):
    if len(re.findall(r'%s' % emoticon_RE_str, comment)) > 0:
        comment = comment + " PUNCxEMOTICON"
    if len(re.findall(r'%s' % exclamation_point_RE_str, comment)) > 0:
        comment = comment + " PUNCxEXCLAMATION_POINT"
    if len(re.findall(r'%s' % question_mark_RE_str, comment)) > 0:
        comment = comment + " PUNCxQUESTION_MARK"
    if len(re.findall(r'%s' % interrobang_RE_str, comment)) > 0:
        comment = comment + " PUNCxINTERROBANG"
    
    if any([len(s) > 2 and str.isupper(s) for s in comment.split(" ")]):
        comment = comment + " PUNCxUPPERCASE" 
    
    comment_texts[i] = comment
# vectorize
vectorizer = CountVectorizer(max_features=50000, ngram_range=(1,2), binary=True)
X = vectorizer.fit_transform(comment_texts)
kf = KFold(len(y), n_folds=5, shuffle=True)
X_context, y_mistakes = [], []
recalls, precisions = [], []
Fs = []
acc=[]
top_features = []
for train, test in kf:
    train_ids = get_entries(all_comment_ids, train)
    test_ids = get_entries(all_comment_ids, test)
    y_train = get_entries(y, train)
    y_test = get_entries(y, test)

    X_train, X_test = X[train], X[test]
    svm = SGDClassifier(loss="hinge", penalty="l2", class_weight="auto", alpha=.01)
    parameters = {'alpha':[.001, .01,  .1]}
    clf = GridSearchCV(svm, parameters, scoring='f1')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc.append(accuracy_score(y_test,clf.predict(X_test)))
    
    #precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_test, preds)
    tp, fp, tn, fn = 0,0,0,0
    N = len(preds)

    for i in xrange(N):
        cur_id = test_ids[i]
        irony_indicator = 1 if cur_id in ironic_comment_ids else 0
        forced_decision_indicator = 1 if cur_id in forced_decision_ids else 0 
        # x1 is the coeffecient for irony (overall)
        # so x2 is the coefficient for forced decisions (i.e., context);
        X_context.append([irony_indicator, forced_decision_indicator])

        y_i = y_test[i]
        pred_y_i = preds[i]

        if y_i == 1:
            # ironic
            if pred_y_i == 1:
                # true positive
                tp += 1 
                y_mistakes.append(0)
            else:
                # false negative
                fn += 1
                y_mistakes.append(1)
        else:
            # unironic
            if pred_y_i == -1:
                # true negative
                tn += 1
                y_mistakes.append(0)
            else:
                # false positive
                fp += 1
                y_mistakes.append(1)

    recall = tp/float(tp + fn)
    precision = tp/float(tp + fp)    
    recalls.append(recall)
    precisions.append(precision)
    f1 = 2* (precision * recall) / (precision + recall)
    Fs.append(f1)

print '\n precisions',precisions
print '\n recalls',recalls
print '\n F1-scores',Fs
print '\n Accuracy of classifier',sum(acc)/5

X_context = sm.add_constant(X_context, prepend=True)
logit_mod = sm.Logit(y_mistakes, X_context)
logit_res = logit_mod.fit()

print logit_res.summary()






