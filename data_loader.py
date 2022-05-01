import warnings
import pickle as pkl
import sys, os

import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np
import pandas as pd

# from sklearn import datasets
# from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from ogb.nodeproppred import DglNodePropPredDataset
# import copy

from utils import sparse_mx_to_torch_sparse_tensor #, dgl_graph_to_torch_sparse

warnings.simplefilter("ignore")


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_citation_network(dataset_str, sparse=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not sparse:
        adj = np.array(adj.todense(),dtype='float32')
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    nfeats = features.shape[1]
    for i in range(labels.shape[0]):
        sum_ = torch.sum(labels[i])
        if sum_ != 1:
            labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
    labels = (labels == 1).nonzero()[:, 1]
    nclasses = torch.max(labels).item() + 1

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj


def load_data(args):
    return load_citation_network(args.dataset, args.sparse)


def load_my_graph(args):
    u = np.loadtxt(os.path.join(args.graphPATH, 'u_24722.txt'))
    v = np.loadtxt(os.path.join(args.graphPATH, 'v_24722.txt'))
    edge_list = np.stack([u, v]).T
    G = nx.Graph()
    G.add_edges_from(edge_list)
    adj = nx.adjacency_matrix(G)
    
    if not args.sparse:
        adj = np.array(adj.todense(),dtype='float32')
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    return adj


def load_my_data(args):
    print('Shape of feature for graph structure learning:', end=' ')
    if args.dataset.upper() == 'UR':
        if args.dataset_type == 'outcode':
            print('Using outpatient diagnoses codes as input features.')
            data_incode = pd.read_csv('../data/data_last_year_out_code.csv')
            print(data_incode.shape)
            print(data_incode.columns)
            data_incode.index = data_incode['No.']
            removed_cancer_cases = {'case_074', 'case_126', 'case_198', 'case_214', 'case_233', 'case_271'}
            X = data_incode.drop(removed_cancer_cases)
            X = X.drop('No.', axis=1)
        elif args.dataset_type == 'pure':
            data = pd.read_csv('../data/data_preprocessed_20220425.csv')
            X = data.drop('target', axis=1)
        y = [1]*301 + [0]*(24722-301)
        y = np.array(y)
        print(X.shape)

    elif 'FH' in args.dataset.upper():
        corpus = pd.read_table('../../FakeHealth/' + args.dataset.upper() + '_corpus.txt', header=None)
        labels = pd.read_table('../../FakeHealth/' + args.dataset.upper() + '_labels.txt', header=None, sep=',')
        enc = TfidfVectorizer(stop_words='english')
        tfidf = enc.fit_transform(corpus.iloc[:,0])
        X = tfidf.todense()
        y = np.array(labels.iloc[:,-1])

    try:
        mask_tr = torch.load('../data/' + args.dataset.upper() + '_mask_tr.pt')
        mask_va = torch.load('../data/' + args.dataset.upper() + '_mask_va.pt')
        mask_te = torch.load('../data/' + args.dataset.upper() + '_mask_te.pt')
    except:
        X = np.arange(tfidf.shape[0])
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, stratify=y, test_size=.3, random_state=args.seed)
        X_va, X_te, y_va, y_te = train_test_split(X_va, y_va, stratify=y_va, test_size=.5, random_state=args.seed)

        print(X_tr.shape, X_va.shape, X_te.shape, y_tr.shape, y_va.shape, y_te.shape)
        
        if args.dataset.upper() == 'UR':
            X = np.array(X)
            idx_tr = [int(i.split('_')[-1]) for i in X_tr.index]
            idx_va = [int(i.split('_')[-1]) for i in X_va.index]
            idx_te = [int(i.split('_')[-1]) for i in X_te.index]
        else:
            X = tfidf.todense()
            idx_tr = list(X_tr)
            idx_va = list(X_va)
            idx_te = list(X_te)
        idx_tr.sort()
        idx_va.sort()
        idx_te.sort()

        print(len(idx_tr), len(idx_va), len(idx_te))
        del X_tr, X_va, X_te
    
        mask_tr = torch.BoolTensor(sample_mask(idx_tr, X.shape[0]))
        mask_va = torch.BoolTensor(sample_mask(idx_va, X.shape[0]))
        mask_te = torch.BoolTensor(sample_mask(idx_te, X.shape[0]))
        torch.save(mask_tr, '../data/' + args.dataset.upper() + '_mask_tr.pt')
        torch.save(mask_va, '../data/' + args.dataset.upper() + '_mask_va.pt')
        torch.save(mask_te, '../data/' + args.dataset.upper() + '_mask_te.pt')

        print(type(X))
        print(X.shape)
    X = torch.FloatTensor(np.array(X))
    Y = torch.LongTensor(y)
    print(X.shape)
    nfeats = X.shape[1]
    nclasses = torch.max(Y).item() + 1

    if args.graphPATH and args.gsl_mode == 'structure_refinement':
        print('structure_refinement')
        adj = load_my_graph(args)
    else:
        adj = torch.eye(X.shape[0])

    return X, nfeats, Y, nclasses, mask_tr, mask_va, mask_te, adj

