worker to tasks preference
CUDA_VISIBLE_DEVICES=0 python Stage1-Predict/worker_to_tasks_train.py --data-train data-process/Foursquare/Foursquare_train.txt --data-val data-process/Foursquare/Foursquare_test.txt --time-units 48 --time-embed-dim 32 --cat-embed-dim 32 --node-attn-nhid 128 --transformer-nhid 1024 --transformer-nlayers 2 --transformer-nhead 2 --batch 16 --epochs 10 --name exp

task to worker preference
CUDA_VISIBLE_DEVICES=9 python task_to_worker_train.py --data-train Foursquare/Foursquare_train.txt --data-val Foursquare/Foursquare_test.txt --time-units 48 --time-embed-dim 32 --cat-embed-dim 32 --node-attn-nhid 128 --transformer-nhid 1024 --transformer-nlayers 2 --transformer-nhead 2 --batch 16 --epochs 10 --name exp



# Foursquare
CUDA_VISIBLE_DEVICES=7 python Stage1-Predict/worker_to_tasks_train_copy.py

CUDA_VISIBLE_DEVICES=5 python Stage1-Predict/task_to_worker_train_copy.py

CUDA_VISIBLE_DEVICES=7 python Stage1-Predict/worker_to_tasks_train_copy.py  --time-units 48 --time_embed_dim 32 --cat_embed_dim 32 --node-attn-nhid 128 --transformer-nhid 1024 --transformer-nlayers 2 --transformer-nhead 2 --batch 16 --epochs 10 --name exp-t2w --data-train data-process/Foursquare/Foursquare_train.txt --data-val data-process/Foursquare/Foursquare_test.txt --task_data_adj_mtx data-process/Foursquare/task_graph_A.csv --task_data_node_feats data-process/Foursquare/task_graph_X.csv --worker_data_adj_mtx data-process/Foursquare/worker_graph_A.csv --worker_data_node_feats data-process/Foursquare/worker_graph_X.csv

# Yelp
CUDA_VISIBLE_DEVICES=5 python Stage1-Predict/worker_to_tasks_train_copy.py 

CUDA_VISIBLE_DEVICES=5 python Stage1-Predict/worker_to_tasks_train_copy.py  --data-train data-process/Yelp_FGRec/Yelp_train.txt --data-val data-process/Yelp_FGRec/Yelp_test.txt --task_data-adj-mtx data-process/Yelp_FGRec/task_graph_A.csv --task_data-node-feats data-process/Yelp_FGRec/task_graph_X.csv --worker_data-adj-mtx data-process/Yelp_FGRec/worker_graph_A.csv --worker_data-node-feats data-process/Yelp_FGRec/worker_graph_X.csv --time-units 48 --time-embed-dim 32 --cat-embed-dim 256 --node-attn-nhid 128 --transformer-nhid 1024 --transformer-nlayers 2 --transformer-nhead 2 --batch 16 --epochs 10 --name exp-w2t

CUDA_VISIBLE_DEVICES=5 python Stage1-Predict/task_to_worker_train_copy.py --data-train data-process/Yelp_FGRec/Yelp_train.txt --data-val data-process/Yelp_FGRec/Yelp_test.txt --task_data-adj-mtx data-process/Yelp_FGRec/task_graph_A.csv --task_data-node-feats data-process/Yelp_FGRec/task_graph_X.csv --worker_data-adj-mtx data-process/Yelp_FGRec/worker_graph_A.csv --worker_data-node-feats data-process/Yelp_FGRec/worker_graph_X.csv --time-units 48 --time-embed-dim 32 --cat-embed-dim 256 --node-attn-nhid 128 --transformer-nhid 1024 --transformer-nlayers 2 --transformer-nhead 2 --batch 16 --epochs 10 --name exp-t2w

