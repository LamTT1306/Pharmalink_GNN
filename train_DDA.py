import timeit
import argparse
import os
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.optim.swa_utils import AveragedModel
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
    parser.add_argument('--fuzzy_proj_dim', type=int, default=64,
                        help='projection dim inside LearnableFuzzyLayer (prevents T-norm collapse)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='label smoothing for CrossEntropyLoss (e.g. 0.05 for gnn_fuzzy)')
    parser.add_argument('--fuzzy_warmup', type=int, default=0,
                        help='epochs to freeze fuzzy layer while backbone warms up (gnn_fuzzy only; 0=no warmup)')
    parser.add_argument('--ortho_weight', type=float, default=0.0005,
                        help='weight of orthogonal diversity loss on fuzzy centers (gnn_fuzzy only)')
    parser.add_argument('--fuzzy_lr_ratio', type=float, default=0.1,
                        help='LR of fuzzy params = backbone LR * fuzzy_lr_ratio (soft unfreeze, prevents gradient shock)')
    parser.add_argument('--fuzzy_dropout', type=float, default=0.05,
                        help='dropout rate on normalised firing strengths inside LearnableFuzzyLayer')
    parser.add_argument('--sparse_weight', type=float, default=0.0,
                        help='weight of L1 sparsity loss on pre-norm firing strengths (gnn_fuzzy only)')
    parser.add_argument('--swa_start_ratio', type=float, default=0.7,
                        help='SWA starts at this fraction of total epochs (e.g. 0.7 = epoch 700/1000)')
    # Vấn đề 3: Chunked gradient accumulation để tránh OOM khi is_finetuning=True
    parser.add_argument('--chunk_size', type=int, default=1024,
                        help='số sample mỗi mini-chunk khi chunked grad accumulation (gnn_fuzzy only)')
    parser.add_argument('--suspect_score_thr', type=float, default=0.25,
                        help='chỉ cách ly cặp nghi vấn có combined_score >= ngưỡng này khi training '
                             '(0.25=an toàn cho dataset có Tanimoto thấp; tăng lên 0.3 nếu muốn cách ly ít hơn)')

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

    cross_entropy = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, 'label_smoothing', 0.0))

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
        if args.model == 'gnn_fuzzy':
            # ── Fix 2: Differential LR — fuzzy params get smaller LR so
            #   backbone gradient is not shocked when fuzzy unfreezes.
            fuzzy_param_ids = set()
            for m in [model.fuzzy_layer, model.fuzzy_correction]:
                fuzzy_param_ids.update(id(p) for p in m.parameters())
            backbone_params = [p for p in model.parameters() if id(p) not in fuzzy_param_ids]
            fuzzy_params    = [p for p in model.parameters() if id(p) in fuzzy_param_ids]
            fuzzy_lr = args.lr * getattr(args, 'fuzzy_lr_ratio', 0.1)
            optimizer = optim.Adam([
                {'params': backbone_params, 'lr': args.lr},
                {'params': fuzzy_params,    'lr': fuzzy_lr},
            ], weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        # Cả 2 model đều dùng fixed LR như paper gốc.
        # ReduceLROnPlateau bị loại bỏ vì nó giảm LR xuống min_lr trước khi SWA
        # kịp hoạt động (SWA cần LR đủ lớn để model dao động trong loss valley).
        scheduler = None

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        best_epoch = 0

        # ── SWA setup (gnn_fuzzy only) ────────────────────────────────
        # AveragedModel smooths out noise when model oscillates in loss valley.
        # No update_bn needed — model uses LayerNorm, not BatchNorm.
        use_swa = (args.model == 'gnn_fuzzy')
        if use_swa:
            swa_model  = AveragedModel(model)
            swa_start  = int(args.epochs * getattr(args, 'swa_start_ratio', 0.7))
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            # ── Cách 3: Two-stage warm-up ─────────────────────────
            # Freeze fuzzy layer for first `fuzzy_warmup` epochs so the backbone
            # (GT + HGT + Transformer) stabilises before fuzzy rules start learning.
            fuzzy_active = True
            if args.model == 'gnn_fuzzy':
                warmup = getattr(args, 'fuzzy_warmup', 10)
                fuzzy_active = (epoch >= warmup)
                for p in model.fuzzy_layer.parameters():
                    p.requires_grad_(fuzzy_active)
                # Also freeze correction projection so backbone trains cleanly
                if hasattr(model, 'fuzzy_correction'):
                    model.fuzzy_correction.requires_grad_(fuzzy_active)

            model.train()

            if args.model == 'gnn_fuzzy':
                # ── Vấn đề 3: Chunked gradient accumulation ──────────────
                # GNN (gt_drug, gt_disease, hgt, transformer fusion) chạy MỘT LẦN
                # để tạo dr_all, di_all. Sau đó chunk X_train theo sample_level
                # (interact → fuzzy → mlp) để tránh OOM khi is_finetuning=True.
                # retain_graph=True giữ GNN graph cho đến chunk cuối cùng.
                CHUNK = args.chunk_size
                dr_all, di_all = model._get_embeddings(
                    drdr_graph, didi_graph, drdipr_graph,
                    drug_feature, disease_feature, protein_feature)

                chunks_x  = torch.split(X_train, CHUNK)
                chunks_y  = torch.split(torch.flatten(Y_train), CHUNK)
                n_chunks  = len(chunks_x)

                optimizer.zero_grad()
                train_loss = 0.0
                for c_idx, (x_c, y_c) in enumerate(zip(chunks_x, chunks_y)):
                    is_last = (c_idx == n_chunks - 1)

                    # Cải tiến C: Bilinear pair — khớp với _get_interact() trong model
                    dr_s = dr_all[x_c[:, 0]]
                    di_s = di_all[x_c[:, 1]]
                    interact   = model.pair_proj(
                        torch.cat([dr_s, di_s, dr_s * di_s], dim=-1))   # (B, 400)
                    # Vấn đề 1: is_finetuning=True → tháo .detach() để co-adaptation
                    fuzzy_in   = interact if fuzzy_active else interact.detach()
                    fuzzy_out  = model.fuzzy_layer(fuzzy_in)
                    correction = model.fuzzy_correction(fuzzy_out)
                    enhanced   = interact + correction
                    c_score    = model.mlp(enhanced)

                    c_loss = cross_entropy(c_score, y_c) / n_chunks

                    # Ortho loss: chỉ cộng vào chunk cuối (không phụ thuộc batch)
                    if getattr(args, 'ortho_weight', 0.0) > 0 and fuzzy_active and is_last:
                        C      = model.fuzzy_layer.centers
                        C_norm = fn.normalize(C, dim=-1)
                        R_size = C_norm.shape[0]
                        sim_mat = C_norm @ C_norm.T
                        eye    = torch.eye(R_size, device=C.device)
                        c_loss = c_loss + args.ortho_weight * (sim_mat - eye).pow(2).mean()

                    # Sparsity loss: cộng vào mỗi chunk, chia đều
                    if getattr(args, 'sparse_weight', 0.0) > 0 and fuzzy_active:
                        raw_w = getattr(model.fuzzy_layer, '_last_raw_w', None)
                        if raw_w is not None:
                            c_loss = c_loss + (args.sparse_weight * raw_w.mean()) / n_chunks

                    # retain_graph cho đến chunk cuối để GNN graph không bị giải phóng sớm
                    c_loss.backward(retain_graph=not is_last)
                    train_loss += c_loss.item()

            else:
                # ── AMNTDDA gốc: full-batch như cũ ───────────────────────
                _, train_score = model(drdr_graph, didi_graph, drdipr_graph,
                                       drug_feature, disease_feature, protein_feature,
                                       X_train)
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

            # ── SWA: accumulate weights + evaluate averaged model ─────────
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                with torch.no_grad():
                    swa_model.eval()
                    _, swa_score = swa_model(
                        drdr_graph, didi_graph, drdipr_graph,
                        drug_feature, disease_feature, protein_feature, X_test)
                swa_prob = fn.softmax(swa_score, dim=-1)[:, 1].cpu().numpy()
                swa_pred = torch.argmax(swa_score, dim=-1).cpu().numpy()
                swa_auc, swa_aupr, swa_acc, swa_prec, swa_rec, swa_f1, swa_mcc = \
                    get_metric(Y_test, swa_pred, swa_prob)
                if swa_auc > best_auc:
                    best_epoch = epoch + 1
                    best_auc = swa_auc
                    best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = \
                        swa_aupr, swa_acc, swa_prec, swa_rec, swa_f1, swa_mcc
                    print(f'AUC improved (SWA) at epoch {best_epoch} ;\tbest_auc: {best_auc}')
                    os.makedirs('web_app/models', exist_ok=True)
                    torch.save({
                        'model_state_dict': swa_model.module.state_dict(),
                        'model_type': args.model,
                        'fold': i,
                        'epoch': best_epoch,
                        'auc': best_auc,
                        'args': args,
                    }, args.checkpoint_path)
                    import shutil
                    shutil.copy2(args.checkpoint_path, args.checkpoint_path_legacy)

            # ── ReduceLROnPlateau — step with current AUC (gnn_fuzzy only)
            if scheduler is not None:
                scheduler.step(AUC)

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
            'Best_Epoch': best_epoch,
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



