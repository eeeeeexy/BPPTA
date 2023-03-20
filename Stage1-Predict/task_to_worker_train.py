import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_worker_graph_node_features
from model import GCN, NodeAttnMap, TaskEmbeddings, UserEmbeddings, WorkerFuseEmbeddings, TaskFuseEmbeddings, WorkerTransformerModel
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep



from param_parser import parameter_parser

def Tasktrain(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)

    # Build POI graph (built from train_df)
    print('Loading worker graph...')
    raw_A = load_graph_adj_mtx(args.worker_data_adj_mtx)
    raw_X = load_worker_graph_node_features(args.worker_data_node_feats,
                                     args.feature3,
                                     args.feature4)

    num_users = raw_X.shape[0]

    X = np.zeros((num_users, raw_X.shape[-1] - 1 + num_users), dtype=np.float32)
    X = raw_X
    
    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

    # POI id to index``  Worker
    nodes_df = pd.read_csv(args.worker_data_node_feats)
    user_ids = list(set(nodes_df['node_name/worker_id'].tolist()))
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))
        
    # User id to index  Task
    task_ids = [each for each in list(set(train_df['poi_id'].to_list()))]
    task_id2idx_dict = dict(zip(task_ids, range(len(task_ids))))

    print(f'users: {len(user_id2idx_dict)}')

    # Print user-trajectories count
    # traj_list = list(set(train_df['trajectory_id'].tolist()))

        # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df['worker_trajectory_id'].tolist())):
                traj_df = train_df[train_df['worker_trajectory_id'] == traj_id]
                user_ids = traj_df['user_id'].to_list()
                user_idxs = [user_id2idx_dict[each] for each in user_ids]

                input_seq = []
                label_seq = []
                for i in range(len(user_idxs) - 1):
                    input_seq.append(user_idxs[i])
                    label_seq.append(user_idxs[i + 1])

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            task_list = []

            for traj_id in tqdm(set(df['worker_trajectory_id'].tolist())):
                task_id = traj_id.split('_')[0]
    
                # Ignore user if not in training set
                if int(task_id) not in task_id2idx_dict.keys():
                    print(f'{task_id} not in task_id2idx_dict ~~~')
                    continue

                # Ger POIs idx in this trajectory
                traj_df = df[df['worker_trajectory_id'] == traj_id]
                user_ids = traj_df['user_id'].to_list()
                user_idxs = []

                for each in user_ids:
                    if each in user_id2idx_dict.keys():
                        user_idxs.append(user_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(user_idxs) - 1):
                    input_seq.append((user_idxs[i]))
                    label_seq.append((user_idxs[i + 1]))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue
                else:
                    task_list.append(task_id)

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)
            
            print(f'task list: {task_list}')
            print(f'task list len: {len(task_list)}')

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])


    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    print(f'train_dataset: {len(train_dataset.input_seqs)}')

    import pdb; pdb.set_trace()

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # print(f'val shape: {val_loader.shape}')

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    args.gcn_nfeat = X.shape[1]
    user_embed_model = GCN(ninput=args.gcn_nfeat,
                          nhid=args.gcn_nhid,
                          noutput=args.user_embed_dim,
                          dropout=args.gcn_dropout)

    # Node Attn Model
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)

    # %% Model2: User embedding model, nn.embedding
    num_tasks = len(task_id2idx_dict)
    task_embed_model = TaskEmbeddings(num_tasks, args.task_embed_dim)

    # %% Model5: Embedding fusion models
    embed_fuse_model1 = WorkerFuseEmbeddings(args.user_embed_dim, args.task_embed_dim)

    # %% Model6: Sequence model
    args.seq_input_embed = args.user_embed_dim + args.task_embed_dim
    seq_model = WorkerTransformerModel(num_users,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)

    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(task_embed_model.parameters()) +
                                  list(node_attn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    # criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)
    
    # %% Tool functions for training
    def input_traj_to_embeddings(sample, task_embeddings):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each for each in sample[1]]

        # User to embedding
        task_id = traj_id.split('_')[0]
        task_idx = task_id2idx_dict[int(task_id)]
        input = torch.LongTensor([int(task_idx)]).to(device=args.device)
        task_embedding = task_embed_model(input)
        task_embedding = torch.squeeze(task_embedding)

        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            user_embedding = user_embeddings[input_seq[idx]]
            user_embedding = torch.squeeze(user_embedding).to(device=args.device)

            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(task_embedding, user_embedding)

            # Save final embed
            input_seq_embed.append(fused_embedding1)

        return input_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        # print(f'y_pred_poi_adjusted: {y_pred_poi_adjusted.shape}')
        attn_map = node_attn_model(X, A)

        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted


    # %% ====================== Train ======================
    task_embed_model = task_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    # train_epochs_time_loss_list = []
    # train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    # val_epochs_time_loss_list = []
    # val_epochs_cat_loss_list = []

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        task_embed_model.train()
        node_attn_model.train()
        user_embed_model.train()
        embed_fuse_model1.train()
        seq_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        # train_batches_time_loss_list = []
        # train_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)

        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            # batch_seq_labels_time = []
            # batch_seq_labels_cat = []

            batch_traj_id = []

            user_embeddings = user_embed_model(X, A)

            # Convert input seq to embeddings
            for sample in batch:

                # print(f'sample: {sample}')

                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                input_seq = [each for each in sample[1]]
                label_seq = [each for each in sample[2]]

                # print(f'traj_id: {traj_id}')
                # print(f'input_seq: {input_seq}')
                # print(f'label_seq: {label_seq}')
                
                # label_seq_time = [each for each in sample[2]]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, user_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_traj_id.append(traj_id)
            
            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            # label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            # y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_pred_poi = seq_model(x, src_mask)

            # print(f'train y_pred_poi shape: {y_pred_poi.shape}')

            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)

            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            # loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)

            # Final loss
            loss = loss_poi
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            # batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len, traj_id in zip(batch_label_pois, batch_pred_pois, batch_seq_lens, batch_traj_id):
                # print(f'train label_pois: {label_pois.shape, label_pois}')

                label_pois = label_pois[:seq_len]  # shape: (seq_len, )\
                # print(f'train pred pois:', pred_pois.shape)
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)

                # print(f'train pred pois: {pred_pois.shape}')
                top1_acc += top_k_acc_last_timestep(label_pois, traj_id, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, traj_id, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, traj_id, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, traj_id, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            # train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            
            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}')
                logging.info(f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}')
                            #  f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                logging.info(f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}')
                logging.info(f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}')
                logging.info(f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}')
                logging.info(f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}')
                logging.info(f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}')
                logging.info(f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}')
                logging.info(f'traj_id:{batch[sample_idx][0]}')
                logging.info(f'input_seq: {batch[sample_idx][1]}')
                logging.info(f'label_seq:{batch[sample_idx][2]}')
                logging.info(f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])}')
                logging.info(f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])}\n'
                            + '=' * 100)
                # logging.info(f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])}\n'
                            #  f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                            #  f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                            #  )
        # train end --------------------------------------------------------------------------------------------------------
        
        preference_poi_array = []
        val_count = 0
        user_traj_array = []

        user_embed_model.eval()
        node_attn_model.eval()
        task_embed_model.eval()
        # time_embed_model.eval()
        embed_fuse_model1.eval()
        # embed_fuse_model2.eval()
        seq_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        # val_batches_time_loss_list = []
        # val_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)

        # import pdb; pdb.set_trace()

        for vb_idx, batch in enumerate(val_loader):
            val_count += 1
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            # batch_seq_labels_time = []
            # batch_seq_labels_cat = []

            batch_traj_id = []
            batch_input_traj = []
            batch_label_traj = []

            user_embeddings = user_embed_model(X, A)

            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each for each in sample[1]]
                label_seq = [each for each in sample[2]]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, user_embeddings))
                
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))

                batch_traj_id.append(traj_id)
                batch_input_traj.append(input_seq)
                batch_label_traj.append(label_seq)

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_pred_poi = seq_model(x, src_mask)

            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)

            # Calculate loss
            loss = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len, traj_id, traj_single, label_single in zip(batch_label_pois, batch_pred_pois, batch_seq_lens, batch_traj_id, batch_input_traj, batch_label_traj):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)

                preference_poi_array.append((traj_id, pred_pois[-1].tolist()))
                user_traj_array.append((traj_id, traj_single, label_single, label_pois[-1]))
                
                top1_acc += top_k_acc_last_timestep(label_pois, traj_id, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, traj_id, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, traj_id, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, traj_id, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (args.batch) * 4) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, ')
                logging.info(f'val_batch_loss:{loss.item():.2f}, ')
                logging.info(f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, ')
                logging.info(f'val_move_loss:{np.mean(val_batches_loss_list):.2f}')
                logging.info(f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f}')
                            #  f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                logging.info(f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f}')
                logging.info(f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f}')
                logging.info(f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f}')
                logging.info(f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f}')
                logging.info(f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f}')
                logging.info(f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f}')
                logging.info(f'traj_id:{batch[sample_idx][0]}')
                logging.info(f'input_seq:{batch[sample_idx][1]}')
                logging.info(f'label_seq:{batch[sample_idx][2]}')
                logging.info(f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])}')
                logging.info(f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])}'
                            + '=' * 100)
                # logging.info(f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                            #  f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                            #  f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' 
                            #  )
        
                
        with open('Preference_value/task_to_worker_preference.pkl', 'wb') as f:
            pickle.dump(preference_poi_array, f)
        with open('Preference_value/worker_trajs.pkl', 'wb') as f2:
            pickle.dump(user_traj_array, f2)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        # epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        # epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        # epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        # epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        # train_epochs_time_loss_list.append(epoch_train_time_loss)
        # train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        # val_epochs_time_loss_list.append(epoch_val_time_loss)
        # val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                    #  f"train_time_loss:{epoch_train_time_loss:.4f}, "
                    #  f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                    #  f"val_time_loss: {epoch_val_time_loss:.4f}, "
                    #  f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")


if __name__ == '__main__':
    args = parameter_parser()

     # The name of node features in task_graph_X.csv
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'

    Tasktrain(args)