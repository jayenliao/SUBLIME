import os, argparse, copy, random, time
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import device
from tqdm import trange

import numpy as np
import torch, dgl
import torch.nn.functional as F
from yaml import parse

from data_loader import load_data, load_my_data
from model import GCN, GCL, basic_model, myGCN, GAT
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

EOS = 1e-10

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()
        self.f1 = 0
        self.NN_based_models = ['gcn', 'gat', 'mlp', 'basic', 'pcgnn']

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)
    

    def loss_cls(self, out_model, model, mask, features, labels, device):
        if out_model == 'gcn':
            logits = model(features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
            accu = accuracy(logp[mask], labels[mask])
            pred_class = torch.max(logp[mask], 1)[1].cpu()
        elif out_model == 'basic':
            prob = model(self.z2.cuda(device), self.p)
            #torch.save(prob.cpu().detach(), './output/prob_test.pt')
            try:
                loss = nn.BCELoss()(prob[mask], labels[mask].double())
            except:
                print('Fail to compute the loss!!!!')
            try:
                pred_class = (prob.cpu() > .5)*1
                pred_class = pred_class[mask]
            except:
                print('Fail to get pred_class!!!!')
            accu = accuracy_score(labels[mask].cpu(), pred_class)
            
        try:
            prec = precision_score(labels[mask].cpu(), pred_class)
            reca = recall_score(labels[mask].cpu(), pred_class)
            f1 = f1_score(labels[mask].cpu(), pred_class)
        except:
            prec, reca, f1 = 0, 0, 0

        return loss, accu.item(), prec, reca, f1


    def loss_cls_fine_tune(self, mask, features, labels):
        logits = self.gcn(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        pred_class = torch.max(logp[mask], 1)[1].cpu()
        try:
            prec = precision_score(labels[mask].cpu(), pred_class)
            reca = recall_score(labels[mask].cpu(), pred_class)
            f1 = f1_score(labels[mask].cpu(), pred_class)
        except:
            prec, reca, f1 = 0, 0, 0

        return loss, accu.item(), prec, reca, f1


    def loss_gcl(self, model, graph_learner, features, anchor_adj):

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor, args.device)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner, args.device)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        if not args.use_embeddings:
            return loss, learned_adj
        else:
            self.z2 = z2.detach()
            return loss, learned_adj


    def evaluate_adj_by_cls(self, out_model, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):
        class_weight = {0: 1 - 0.01217539, 1: 0.01217539} if args.dataset == 'UR' else None
        print(features.shape)

        if out_model == 'gcn':
            model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls, dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        
        elif out_model == 'gat':
            g = dgl.remove_self_loop(Adj)
            g = dgl.add_self_loop(g)
            heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
            model = GAT(g=g, num_layers=args.num_layerds, in_dim=nfeats, num_hidden=args.num_hidden, num_classes=nclasses, heads=heads, activation=F.elu, feat_drop=args.in_drop, attn_drop=args.attn_drop, negative_slope=args.negative_slope, residual=args.residual)

        elif out_model == 'basic':
            model = basic_model(args.proj_dim, args.eta)

        elif out_model == 'random_forest':
            model = RandomForestClassifier(n_jobs=args.n_jobs)
        
        elif out_model == 'random_forest_w':
            model = RandomForestClassifier(n_jobs=args.n_jobs, class_weight=class_weight)

        elif out_model == 'xgboost':
            model = XGBClassifier(objective='binary:logistic', subsample=.8, verbosity=0, eval='logloss')
        
        elif out_model == 'xgboost_w':
            model = XGBClassifier(objective='binary:logistic', subsample=.8, verbosity=0, eval='logloss', scale_pos_weight=(24722-301)/301)

        elif out_model == 'catboost':
            task_type = 'GPU' if torch.cuda.is_available() and 'cuda' in args.device else 'CPU'
            model = CatBoostClassifier(task_type=task_type, silent=True)

        if out_model in self.NN_based_models:
            best_val, test_accu, test_prec, test_reca, test_f1, best_model = self.evaluation_for_NN(Adj, out_model, model, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args)
        else:
            best_val, test_accu, test_prec, test_reca, test_f1, best_model = self.evaluation_for_nonNN(out_model, model, features, labels, train_mask, val_mask, test_mask)
        
        return best_val, test_accu, test_prec, test_reca, test_f1, best_model


    def evaluation_for_nonNN(self, out_model, model, features, labels, train_mask, val_mask, test_mask):
        # do something here ....
        print('\nTraining', out_model, '...')
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        train_mask = train_mask.cpu().numpy()
        val_mask = val_mask.cpu().numpy()
        test_mask = test_mask.cpu().numpy()

        model.fit(features[train_mask], labels[train_mask])
        y_pred_tr = model.predict(features[train_mask])
        y_pred_va = model.predict(features[val_mask])
        y_pred_te = model.predict(features[test_mask])

        train_results = {'Accuracy': accuracy_score(labels[train_mask], y_pred_tr), 'Precision': precision_score(labels[train_mask], y_pred_tr), 'Recall': recall_score(labels[train_mask], y_pred_tr), 'F1': f1_score(labels[train_mask], y_pred_tr)}
        val_results = {'Accuracy': accuracy_score(labels[val_mask], y_pred_va), 'Precision': precision_score(labels[val_mask], y_pred_va), 'Recall': recall_score(labels[val_mask], y_pred_va), 'F1': f1_score(labels[val_mask], y_pred_va)}
        test_accu = accuracy_score(labels[test_mask], y_pred_te)
        test_prec = precision_score(labels[test_mask], y_pred_te)
        test_reca = recall_score(labels[test_mask], y_pred_te)
        test_f1 = f1_score(labels[test_mask], y_pred_te)
        
        print('[Training]  ', end='\t')
        for k, v in train_results.items():
            print('%s=%.4f' % (k, v), end=' ')
        print('\n[Validation]', end='\t')
        for k, v in val_results.items():
            print('%s=%.4f' % (k, v), end= ' ')
        print('\n[Testing]', end='\t')
        for i, k in enumerate(val_results.keys()):
            v = (test_accu, test_prec, test_reca, test_f1)[i]
            print('%s=%.4f' % (k, v), end= ' ')
        print()

        return val_results, test_accu, test_prec, test_reca, test_f1, None


    def evaluation_for_NN(self, Adj, out_model, model, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}
        best_model = None #copy.deepcopy(model)

        if torch.cuda.is_available():
            model = model.cuda(args.device)
            train_mask = train_mask.cuda(args.device)
            val_mask = val_mask.cuda(args.device)
            test_mask = test_mask.cuda(args.device)
            features = features.cuda(args.device)
            labels = labels.cuda(args.device)
            if out_model == 'basic':
                self.p = torch.tensor(self.p).cuda(args.device)

        progress_bar_gcn = trange(1, args.epochs_cls + 1)
        best_model = copy.deepcopy(model)
        for epoch in progress_bar_gcn:
            model.train()
            loss, accu, prec, reca, f1 = self.loss_cls(out_model, model, train_mask, features, labels, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ = loss.cpu().item()
            self.training_loss_cls[out_model][-1].append(loss_)

            if epoch % 1 == 0:
                model.eval()
                val_loss, accu, prec, reca, self.f1 = self.loss_cls(out_model, model, val_mask, features, labels, args.device)
                if self.f1 > best_val['F1']:
                    bad_counter = 0
                    best_val = {'Accuracy': accu, 'Precision': prec, 'Recall': reca, 'F1': self.f1}
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break
            progress_bar_gcn.set_description('Train loss: %.4f | Val loss: %.4f | Best val Acc=%.4f F1=%.4f' % (loss_, val_loss.item(), best_val['Accuracy'], best_val['F1']))
        
        best_model.eval()
        test_loss, test_accu, test_prec, test_reca, test_f1 = self.loss_cls(out_model, best_model, test_mask, features, labels, args.device)

        return best_val, test_accu, test_prec, test_reca, test_f1, best_model


    def train(self, args):
        print('We are on %s.' % args.device)
        if 'basic' in args.out_models:
            self.p = np.load('../Non-GNN/output/prob_from_catboost_21f.npy')
        torch.cuda.set_device(args.device)

        # Make the folder for outputs files
        self.dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        folder_ = args.dataset + '_s' + args.gsl_mode.split('_')[-1][0]
        folder_ = folder_ + '_' + args.dataset_type if args.dataset == 'UR' else folder_
        folder__ = 'emb' if args.use_embeddings else 'org'
        self.output_folder = os.path.join('./output', folder_, folder__ + '_' + self.dt)
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Load the data
        if args.gsl_mode == 'structure_refinement':
            if args.dataset == 'UR' or 'FH' in args.dataset:
                features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_my_data(args)
            else:
                features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data(args)
        elif args.gsl_mode == 'structure_inference':
            if args.dataset == 'UR' or 'FH' in args.dataset:
                features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = load_my_data(args)
            else:
                features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = load_data(args)

        # Prepare containers for training records
        if args.downstream_task == 'classification':
            test_dicts = []
            validation_dicts = []
        elif args.downstream_task == 'clustering':
            n_clu_trials = copy.deepcopy(args.ntrials)
            args.ntrials = 1

        self.training_loss_gcl = {}
        self.training_metrics = {'Accuracy': {}, 'Precision': {}, 'Recall': {}, 'F1': {}}
        for m in self.training_metrics.keys():
            for out_model in args.out_models:
                self.training_metrics[m][out_model.lower()] = []

        self.training_loss_cls = {}
        for out_model in args.out_models:
            out_model = out_model.lower()
            if out_model in self.NN_based_models:
                self.training_loss_cls[out_model] = []

        # Start to train for arg.ntrials trials !!
        for trial in range(args.ntrials):

            self.setup_seed(trial)
            self.training_loss_gcl[trial] = []

            # Initialize the adjacency matrix
            if args.gsl_mode == 'structure_inference':
                if args.sparse:
                    anchor_adj_raw = torch_sparse_eye(features.shape[0])
                else:
                    anchor_adj_raw = torch.eye(features.shape[0])
            elif args.gsl_mode == 'structure_refinement':
                if args.sparse:
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = torch.from_numpy(adj_original)

            # Normalize the adjacency matrix
            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)
            
            # Sparsilize the adjacency matrix
            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj, args.device)

            # Graph Learner
            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse, args.activation_learner, args.device)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse, args.activation_learner, args.device)
            elif args.type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse, args.activation_learner, anchor_adj, args.device)

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim, emb_dim=args.rep_dim, proj_dim=args.proj_dim, dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)
            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)

            if torch.cuda.is_available():
                model = model.cuda(args.device)
                graph_learner = graph_learner.cuda(args.device)
                train_mask = train_mask.cuda(args.device)
                val_mask = val_mask.cuda(args.device)
                test_mask = test_mask.cuda(args.device)
                features = features.cuda(args.device)
                labels = labels.cuda(args.device)
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda(args.device)

            if args.downstream_task == 'classification':
                best_val = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}
                best_val_test = 0
                best_epoch = 0
                count_outperform = 0

            progress_bar = trange(1, args.epochs + 1)
            for epoch in progress_bar:

                model.train()
                graph_learner.train()
                loss, Adj, = self.loss_gcl(model, graph_learner, features, anchor_adj)
                # self.z2 is computed at this moment

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                #retain_graph = True if args.fine_tune else False
                if args.fine_tune:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

                # Structure Bootstrapping
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse, args.device)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)
                self.training_loss_gcl[trial].append(loss.item())

                # if epoch % 500 == 0:
                #     output_folder = os.path.join('./output', args.dataset)
                #     try:
                #         os.mkdir(output_folder)
                #     except:
                #         pass
                #     np.save(os.path.join(output_folder, self.dt+'_z2.npy'), z2.cpu().detach().numpy())
                
                # Evaluate the graph structure learning by the downstream task
                eval_freq = 1 if args.fine_tune else args.eval_freq
                if epoch % eval_freq == 0:
                    if args.downstream_task == 'classification':
                        for out_model in args.out_models:
                            out_model = out_model.lower()
                            if out_model in self.NN_based_models:
                                self.training_loss_cls[out_model].append([])
                        
                        if args.fine_tune:
                            self.gcn = myGCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls, dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
                            optimizer_gcn = torch.optim.Adam(self.gcn.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)
                            if torch.cuda.is_available():
                                self.gcn = self.gcn.cuda(args.device)
                            self.gcn.train()
                            loss_gcn, accu, prec, reca, f1 = self.loss_cls_fine_tune(train_mask, features, labels)
                            optimizer_gcn.zero_grad()
                            #loss.backward()
                            loss_gcn.backward()
                            optimizer_gcn.step()
                            #optimizer_cl.step()
                            #optimizer_learner.step()
                            
                            self.gcn.eval()
                            best_val = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}
                            val_loss, accu, prec, reca, self.f1 = self.loss_cls_fine_tune(val_mask, features, labels)
                            if f1 > best_val['F1']:
                                bad_counter = 0
                                best_val = {'Accuracy': accu, 'Precision': prec, 'Recall': reca, 'F1': self.f1}
                                print(best_val)
                                #best_model = copy.deepcopy(model)
                            else:
                                bad_counter += 1
                            if bad_counter >= args.patience_cls:
                                break

                        else:
                            model.eval()
                            graph_learner.eval()
                            f_adj = Adj

                            if args.sparse:
                                f_adj.edata['w'] = f_adj.edata['w'].detach()
                            else:
                                f_adj = f_adj.detach()

                            for out_model in args.out_models:
                                out_model = out_model.lower()
                                t_step_0 = time.time() 
                                
                                if out_model in self.NN_based_models and not args.use_embeddings:
                                    val_now, test_accu, test_prec, test_reca, test_f1, best_model = self.evaluate_adj_by_cls(
                                        out_model, f_adj, features, nfeats, labels, nclasses,
                                        train_mask, val_mask, test_mask, args)
                                else:
                                    val_now, test_accu, test_prec, test_reca, test_f1, best_model = self.evaluate_adj_by_cls(
                                        out_model, f_adj, self.z2, self.z2.shape[1], labels, nclasses,
                                        train_mask, val_mask, test_mask, args)

                                for m in self.training_metrics.keys():
                                    self.training_metrics[m][out_model].append(val_now[m])
                                
                                print('Time cost of %s: %.2f' % (out_model, time.time() - t_step_0))
                                
                                if val_now['F1'] > best_val['F1']:
                                    best_val = val_now.copy()
                                    best_val_test = {'Accuracy': test_accu, 'Precision': test_prec, 'Recall': test_reca, 'F1': test_f1}
                                    best_epoch = epoch
                                    if args.use_embeddings and trial == 0:
                                        np.save(os.path.join(self.output_folder, 'z2.npy'), self.z2.cpu().detach().numpy())
                                    count_outperform += 1
                                #torch.save(best_model, './output/' + self.dt + '_best_model.pt')
                                #np.save('./output/' + self.dt + '_f_adj.npy', f_adj)
                                #torch.save(model, './output/' + self.dt + '_model.pt')

                    elif args.downstream_task == 'clustering':
                        model.eval()
                        graph_learner.eval()
                        _, embedding = model(features, Adj)
                        embedding = embedding.cpu().detach().numpy()

                        acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []
                        for clu_trial in range(n_clu_trials):
                            kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial).fit(embedding)
                            predict_labels = kmeans.predict(embedding)
                            cm_all = clustering_metrics(labels.cpu().numpy(), predict_labels)
                            acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                            acc_mr.append(acc_)
                            nmi_mr.append(nmi_)
                            f1_mr.append(f1_)
                            ari_mr.append(ari_)

                        acc, nmi, f1, ari = np.mean(acc_mr), np.mean(nmi_mr), np.mean(f1_mr), np.mean(ari_mr)

                if args.fine_tune:
                    progress_bar.set_description('[Trial %d Epoch %4d] CL loss=%.4f GCN loss=%.4f' % (trial, epoch, loss.item(), loss_gcn.item()))
                else:
                    progress_bar.set_description('[Trial %d Epoch %4d] CL loss=%.4f' % (trial, epoch, loss.item()))


            if args.downstream_task == 'classification':
                print("Trial %d has been completed." % (trial+1))
                validation_dicts.append(best_val)
                test_dicts.append(best_val_test)
                print("Best val (epoch %d):" % best_epoch, end=' ')
                print(best_val)
                print("Best test (epoch %d):" % best_epoch, end=' ')
                print(best_val_test)

            elif args.downstream_task == 'clustering':
                print("Final ACC: ", acc)
                print("Final NMI: ", nmi)
                print("Final F-score: ", f1)
                print("Final ARI: ", ari)
        
        for out_model in args.out_models:
            out_model = out_model.lower()
            if out_model in self.NN_based_models:
                self.plot_training('loss_cls', figsize=args.figsize, saveplot=args.saveplot, out_model=out_model)

        if args.downstream_task == 'classification' and trial != 0:
            for m in best_val.keys():
                self.print_results(validation_dicts, test_dicts, m)

    def print_results(self, validation_dicts, test_dicts, metric):
        validation_m = [v[metric] for v in validation_dicts]
        test_m = [v[metric] for v in test_dicts]
        s_val = "Val {:s}: {:.4f} +/- {:.4f}".format(metric, np.mean(validation_m), np.std(validation_m))
        s_test = "Test {:s}: {:.4f} +/- {:.4f}".format(metric, np.mean(test_m),np.std(test_m))
        print(s_val)
        print(s_test)
        print()

    def plot_training(self, metric, figsize, saveplot=True, out_model=None):
        plt.figure(figsize=figsize)
        
        if metric == 'loss_gcl':
            dy = self.training_loss_gcl
            title_ = 'Training Loss of Graph Structure Learning'
            ylab = 'GCL Loss'
        elif metric in ['loss_gcn', 'loss_cls']:
            dy = {}
            for k, arr in enumerate(self.training_loss_cls[out_model]):
                dy[k] = arr
            title_ = 'Training Loss of Downstream Model: ' + out_model.upper()
            ylab = 'NLL Loss'
        # else:
        #     dy = self.training_metrics
        #     ylab = metric.capitalize()
        #     title_ = 'Validation ' + ylab + ' during Training'
        #     plt.ylim((0, 1))

        for trial, y in dy.items():
            x = np.arange(len(y))
            plt.plot(x, y, label=trial)
            plt.title(title_)
            plt.xlabel('Epoch')
            plt.ylabel(ylab)
            plt.legend()
            plt.grid()
        
        if saveplot:
            if out_model:
                fn = os.path.join(self.output_folder, metric + '_' + out_model + '.png')
            else:
                fn = os.path.join(self.output_folder, metric + '.png')
            plt.savefig(fn)
            print('--->', fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'UR', 'FH', 'FHR', 'FHS'])
    parser.add_argument('-dataset_type', type=str, default='pure')
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference", choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='classification', choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-graphPATH', type=str, default='../eICU-GNN-LSTM/graph/')
    parser.add_argument('-seed', type=int, default=4028)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-out_models', nargs='+', type=str, default=['gcn'])
    parser.add_argument('-eta', type=float, default=.9)
    parser.add_argument('-device', type=str, default='cuda:0')
    parser.add_argument('-use_embeddings', action='store_true')
    parser.add_argument('-fine_tune', action='store_true')

    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    # GAT
    parser.add_argument("-num-heads", type=int, default=3, help="number of hidden attention heads")
    parser.add_argument("-num-out-heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("-num-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("-num-hidden", type=int, default=8, help="number of hidden units")
    parser.add_argument("-residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("-in_drop", type=float, default=.6, help="input feature dropout")
    parser.add_argument("-attn_drop", type=float, default=.6, help="attention dropout")
    parser.add_argument('-weight-decay', type=float, default=5e-4,  help="weight decay")
    parser.add_argument('-negative-slope', type=float, default=0.2, help="the negative slope of leaky relu")
    # parser.add_argument("-lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('-early-stop', action='store_true', default=False, help="indicates whether to use early stop or not")
    parser.add_argument('-fastmode', action="store_true", default=False, help="skip re-evaluate the validation set")

    # Boosting
    parser.add_argument('-n_jobs', type=int, default=8)

    # Plot results
    parser.add_argument('-figsize', type=tuple, default=(8,6))
    parser.add_argument('-saveplot', action='store_true')

    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args)
    experiment.plot_training('loss_gcl', figsize=args.figsize, saveplot=args.saveplot)
    
    with open(os.path.join(experiment.output_folder, 'training_metrics.pkl'), 'wb') as f:
        pkl.dump(experiment.training_metrics, f, pkl.HIGHEST_PROTOCOL)
    
    with open(os.path.join(experiment.output_folder, 'training_loss_gcl.pkl'), 'wb') as f:
        pkl.dump(experiment.training_loss_gcl, f, pkl.HIGHEST_PROTOCOL)
    
    np.save(os.path.join(experiment.output_folder, 'training_loss_cls.npy'), np.array(experiment.training_loss_cls))
    
    #for m in ['Accuracy', 'Precision', 'Recall', 'F1']:
    #    experiment.plot_training(metric=m, figsize=args.figsize, saveplot=args.saveplot)