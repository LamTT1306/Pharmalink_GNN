import timeit
import argparse
import os
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from model.AMNTDDA import AMNTDDA
from model.AMNTDDA_Fuzzy import AMNTDDA_Fuzzy
from metric import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='200', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')
    # Model selection
    parser.add_argument('--model', default='gnn', choices=['gnn', 'gnn_fuzzy'],
                        help='gnn = original AMNTDDA  |  gnn_fuzzy = AMNTDDA + Learnable Fuzzy Layer')
    # Fuzzy layer hyper-parameters (only used when --model gnn_fuzzy)
    parser.add_argument('--fuzzy_rules', type=int, default=32,
                        help='number of TSK fuzzy rules in LearnableFuzzyLayer')
    parser.add_argument('--fuzzy_dim', type=int, default=256,
                        help='output dimension of LearnableFuzzyLayer')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    model_tag = 'AMNTDDA_Fuzzy' if args.model == 'gnn_fuzzy' else 'AMNTDDA'
    args.result_dir = 'Result/' + args.dataset + '/' + model_tag + '/'
    # Checkpoint path: dataset-specific to avoid overwriting models across datasets
    dataset_tag = args.dataset.replace('-', '_').lower()   # e.g. b_dataset, c_dataset
    args.checkpoint_path = (
        f'web_app/models/best_model_fuzzy_{dataset_tag}.pt'
        if args.model == 'gnn_fuzzy'
        else f'web_app/models/best_model_{dataset_tag}.pt'
    )
    # Also keep legacy filenames as symlinks/copies for backward compat with the web app
    args.checkpoint_path_legacy = (
        'web_app/models/best_model_fuzzy.pt'
        if args.model == 'gnn_fuzzy'
        else 'web_app/models/best_model.pt'
    )
    print(f'[train] Model type  : {model_tag}')
    print(f'[train] Checkpoint  : {args.checkpoint_path}')

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []
    fold_results = []   # collect per-fold best metrics for CSV

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        if args.model == 'gnn_fuzzy':
            model = AMNTDDA_Fuzzy(args)
        else:
            model = AMNTDDA(args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            model.train()
            _, train_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train)
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_representation, test_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_test)

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)
                os.makedirs('web_app/models', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_type': args.model,
                    'fold': i,
                    'epoch': best_epoch,
                    'auc': best_auc,
                    'args': args,
                }, args.checkpoint_path)
                # Also write to legacy path so web app always finds a working model
                import shutil
                shutil.copy2(args.checkpoint_path, args.checkpoint_path_legacy)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        fold_results.append({
            'Fold':      i,
            'Best_Epoch': best_epoch if 'best_epoch' in dir() else 0,
            'AUC':       best_auc,
            'AUPR':      best_aupr,
            'Accuracy':  best_accuracy,
            'Precision': best_precision,
            'Recall':    best_recall,
            'F1-score':  best_f1,
            'Mcc':       best_mcc,
        })

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')

    # ── Save 10-fold results CSV (like AMDGT format) ──────────────────────
    import pandas as pd
    from datetime import datetime
    df = pd.DataFrame(fold_results)
    mean_row = {'Fold': 'Mean', 'Best_Epoch': '',
                'AUC':       df['AUC'].mean(),
                'AUPR':      df['AUPR'].mean(),
                'Accuracy':  df['Accuracy'].mean(),
                'Precision': df['Precision'].mean(),
                'Recall':    df['Recall'].mean(),
                'F1-score':  df['F1-score'].mean(),
                'Mcc':       df['Mcc'].mean()}
    std_row  = {'Fold': 'Std',  'Best_Epoch': '',
                'AUC':       df['AUC'].std(),
                'AUPR':      df['AUPR'].std(),
                'Accuracy':  df['Accuracy'].std(),
                'Precision': df['Precision'].std(),
                'Recall':    df['Recall'].std(),
                'F1-score':  df['F1-score'].std(),
                'Mcc':       df['Mcc'].std()}
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    os.makedirs(args.result_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.result_dir, f'10_fold_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Results saved to: {csv_path}')



