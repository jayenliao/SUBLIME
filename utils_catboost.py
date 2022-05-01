import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

def data_transform_lr_cv(data, features=['all'], drop_features='No.', y_label='target', test_size=.2, cv=5, random_state=4028):
    
    if 'all' in features:
        data_f = data.copy()
    else:
        data_f = data[features].copy()
    
    try:
        data_f = data_f.drop(drop_features, axis=1)
    except:
        pass
    
    # Data splitting
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=random_state)  
    
    # Construct the classifier
    fold, d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf = 0, {}, {}, {}, {}, {}
    for train_idx, test_idx in sss.split(data_f, data[y_label]):
        d_X_tr[fold] = data_f.iloc[train_idx,:]
        d_X_te[fold] = data_f.iloc[test_idx,:]
        d_y_tr[fold] = data[y_label].iloc[train_idx]
        d_y_te[fold] = data[y_label].iloc[test_idx]
        d_clf[fold] = LogisticRegression(random_state=random_state)
        fold += 1
    
    return d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf
    

def data_transform_rf(data, features=['all'], drop_features='No.', y_label='target', test_size=.2, random_state=4028):
    
    if 'all' in features:
        data_f = data.copy()
    else:
        data_f = data[features].copy()
    
    try:
        data_f = data_f.drop(drop_features, axis=1)
    except:
        pass
    
    if y_label in data_f.columns:
        data_f = data_f.drop(y_label, axis=1)
    
    # Data splitting
    X_tr, X_te, y_tr, y_te = train_test_split(
        data_f, data['target'], stratify=data[y_label],
        test_size=test_size, random_state=random_state)    
    
    # Construct the classifier
    clf_rf = RandomForestClassifier()
    
    return clf_rf, X_tr, X_te, y_tr, y_te


def data_transform_rf_cv(data, features=['all'], drop_features='No.', y_label='target', test_size=.2, cv=5, random_state=4028):
    
    if 'all' in features:
        data_f = data.copy()
    else:
        data_f = data[features].copy()
    
    try:
        data_f = data_f.drop(drop_features, axis=1)
    except:
        pass
    
    if y_label in data_f.columns:
        data_f = data_f.drop(y_label, axis=1)
    
    # Data splitting
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=random_state)  
    
    # Construct the classifier
    fold, d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf_rf = 0, {}, {}, {}, {}, {}
    for train_idx, test_idx in sss.split(data_f, data[y_label]):
        d_X_tr[fold] = data_f.iloc[train_idx,:]
        d_X_te[fold] = data_f.iloc[test_idx,:]
        d_y_tr[fold] = data[y_label].iloc[train_idx]
        d_y_te[fold] = data[y_label].iloc[test_idx]
        d_clf_rf[fold] = RandomForestClassifier(random_state=random_state)
        fold += 1
    
    return d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf_rf


def data_transform_cat(data, features, cat_features, y_label='taget', test_size=.2, random_state=4028, subsample=.9, verbose=False, eval_metric='F1', task_type='CPU', bootstrap_type=''):
    
    # Features
    data_cat = data[features].copy()
    cat_features = list(set(features) & set(cat_features))
    data_cat[cat_features] = data_cat[cat_features].astype('str')
    non_cat_features = list(set(features) - set(cat_features))
    try:
        for col in non_cat_features:
            if sum(np.isnan(data_cat[col])) > 0:
                data_cat.loc[np.isnan(data_cat[col]), col] = np.median(data_cat.loc[-np.isnan(data_cat[col]), col])
    except:
        pass
    
    # Data splitting
    X_tr, X_te, y_tr, y_te = train_test_split(
        data_cat, data['target'], stratify=data['target'],
        test_size=test_size, random_state=random_state)    
    
    # Construct the classifier
    if task_type == 'GPU':
        bootstrap_type = 'Poisson'
    clf_cat = CatBoostClassifier(cat_features=cat_features, subsample=subsample, eval_metric=eval_metric, verbose=verbose,
                                 task_type=task_type, bootstrap_type=bootstrap_type, random_state=random_state)
    
    return clf_cat, X_tr, X_te, y_tr, y_te


def data_transform_cat_cv(data, features, cat_features, y_label='target', test_size=.2, cv=5, random_state=4028, subsample=.9, verbose=False, eval_metric='F1', task_type='CPU', bootstrap_type=''):
    
    # Features
    data_cat = data[features].copy()
    cat_features = list(set(features) & set(cat_features))
    data_cat[cat_features] = data_cat[cat_features].astype('str')
    non_cat_features = list(set(features) - set(cat_features))
    try:
        for col in non_cat_features:
            if sum(np.isnan(data_cat[col])) > 0:
                data_cat.loc[np.isnan(data_cat[col]), col] = np.median(data_cat.loc[-np.isnan(data_cat[col]), col])
    except:
        pass
    
    # Data splitting
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=random_state)
    
    # Construct the classifier
    if task_type == 'GPU':
        bootstrap_type = 'Poisson'
    
    fold, d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf_cat = 0, {}, {}, {}, {}, {}
    for train_idx, test_idx in sss.split(data_cat[features], data[y_label]):
        d_X_tr[fold] = data_cat.iloc[train_idx,:]
        d_X_te[fold] = data_cat.iloc[test_idx,:]
        d_y_tr[fold] = data[y_label].iloc[train_idx]
        d_y_te[fold] = data[y_label].iloc[test_idx]
        d_clf_cat[fold] = CatBoostClassifier(
            cat_features=cat_features, subsample=subsample, eval_metric=eval_metric, verbose=verbose,
            task_type=task_type, bootstrap_type=bootstrap_type, random_state=random_state)
        fold += 1
        
    return d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf_cat


def data_transform_xgb_cv(data, features=['all'], drop_features=['No.'], y_label='target', test_size=.2, cv=5, random_state=4028, subsample=.9):
    
    if 'all' in features:
        data_f = data.copy()
    else:
        data_f = data[features].copy()
    
    try:
        data_f = data_f.drop(drop_features, axis=1)
    except:
        pass
    
    if y_label in data_f.columns:
        data_f = data_f.drop(y_label, axis=1)
    
    # Data splitting
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=random_state)  
    
    # Construct the classifier
    fold, d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf_xgb = 0, {}, {}, {}, {}, {}
    for train_idx, test_idx in sss.split(data_f, data[y_label]):
        d_X_tr[fold] = data_f.iloc[train_idx,:]
        d_X_te[fold] = data_f.iloc[test_idx,:]
        d_y_tr[fold] = data[y_label].iloc[train_idx]
        d_y_te[fold] = data[y_label].iloc[test_idx]
        d_clf_xgb[fold] = XGBClassifier(subsample=subsample, eval_metric='auc')
        fold += 1
    
    return d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf_xgb

class trainer:
    def __init__(self, d_X_tr, d_X_te, d_y_tr, d_y_te, d_clf):
        self.d_X_tr = d_X_tr
        self.d_X_te = d_X_te
        self.d_y_tr = d_y_tr
        self.d_y_te = d_y_te
        self.d_clf = d_clf
        
    def train(self, save_as_file=True, filePATH='./output/', clf_name=''):
        loader = tqdm(range(len(self.d_clf)))
        for fold in loader:
            self.d_clf[fold].fit(self.d_X_tr[fold], self.d_y_tr[fold])
        if save_as_file:
            f = self.d_X_tr[0].shape[0]
            with open('./output/d_clf_' + clf_name + str(f) + 'f.pkl', 'wb') as f:
                pickle.dump(self.d_clf, f)
    
    def evaluate(self, evaluation_set='testing', print_results=True):
        if evaluation_set == 'training':
            d_X_eval = self.d_X_tr
            d_y_eval = self.d_y_tr
        elif evaluation_set == 'testing':
            d_X_eval = self.d_X_te
            d_y_eval = self.d_y_te
            
        self.d_y_pred_label, self.d_y_pred_proba = {}, {}
        self.lst_precision, self.lst_recall = [], []
        self.lst_f1, self.lst_auroc, self.lst_auprc = [], [], []
        self.lst_fpr, self.lst_tpr, self.lst_pre, self.lst_rec = [], [], [], []
        
        for fold in range(len(self.d_clf)):
            self.d_y_pred_label[fold] = self.d_clf[fold].predict(d_X_eval[fold])
            self.d_y_pred_proba[fold] = self.d_clf[fold].predict_proba(d_X_eval[fold])[:,1]
            self.lst_precision.append(precision_score(d_y_eval[fold], self.d_y_pred_label[fold]))
            self.lst_recall.append(recall_score(d_y_eval[fold], self.d_y_pred_label[fold]))
            self.lst_f1.append(f1_score(d_y_eval[fold], self.d_y_pred_label[fold]))
            
            fpr, tpr, thr = roc_curve(d_y_eval[fold], self.d_y_pred_proba[fold])
            self.lst_auroc.append(auc(fpr, tpr))
            self.lst_fpr.append(fpr)
            self.lst_tpr.append(tpr)
            pre, rec, thr = precision_recall_curve(d_y_eval[fold], self.d_y_pred_proba[fold])
            self.lst_auprc.append(auc(rec, pre))
            self.lst_pre.append(pre)
            self.lst_rec.append(rec)
        
        if print_results:
            print('Precision: %4.4f +- %4.4f' % (np.mean(self.lst_precision), np.std(self.lst_precision)))
            print('Recall:    %4.4f +- %4.4f' % (np.mean(self.lst_recall), np.std(self.lst_recall)))
            print('F1-score:  %4.4f +- %4.4f' % (np.mean(self.lst_f1), np.std(self.lst_f1)))
            print('AUROC:     %4.4f +- %4.4f' % (np.mean(self.lst_auroc), np.std(self.lst_auroc)))
            print('AUPRC:     %4.4f +- %4.4f' % (np.mean(self.lst_auprc), np.std(self.lst_auprc)))

def draw_ROC_curve(fpr, tpr, size=5, lw=2):
    plt.figure()
    plt.figure(figsize=(size, size))
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.4f)' % auc(fpr, tpr))
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

def draw_PRC_curve(precision, recall, size=5, lw=2):
    plt.figure()
    plt.figure(figsize=(size, size))
    plt.plot(recall, precision, color='darkorange', lw=lw,
             label='PRC curve (area = %0.4f)' % auc(recall, precision))
    plt.plot([0,1], [1,0], color='navy', lw=lw, linestyle='--')
    plt.xlabel('Recall Score (Sensitivity)')
    plt.ylabel('Precision Score (True Positive Rate)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

def draw_ROC_curve_cv(lst_fpr, lst_tpr, cv=True, size=5, lw=1.5):
    if cv:
        cv = len(lst_fpr)
    
    plt.figure()
    plt.figure(figsize=(size, size))
    
    for fold in range(cv):
        fpr, tpr = lst_fpr[fold], lst_tpr[fold]
        plt.plot(fpr, tpr, lw=lw,
                 label='Fold %d (area=%0.4f)' % (fold+1, auc(fpr, tpr)))
    
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")
    plt.show()

def draw_PRC_curve_cv(lst_pre, lst_rec, cv=True, size=5, lw=1.5):
    if cv:
        cv = len(lst_pre)
    
    plt.figure()
    plt.figure(figsize=(size, size))
    
    for fold in range(cv):
        pre, rec = lst_pre[fold], lst_rec[fold]
        auc_score_ = auc(rec, pre)
        plt.plot(rec, pre, lw=lw,
                 label='Fold %d (area=%0.4f)' % (fold+1, auc_score_))
    
    plt.plot([0,1], [1,0], color='navy', lw=lw, linestyle='--')
    plt.xlabel('Recall Score (Sensitivity)')
    plt.ylabel('Precision Score (True Positive Rate)')
    plt.title('Precision-Recall Curve')
    if auc_score_ > .5:
        plt.legend(loc="lower left")
    else:
        plt.legend(loc='upper right')
    plt.show()