import argparse
import torch.nn.functional as F
import torch.sparse
import numpy as np
import dgl
import random
import time
from sklearn.model_selection import train_test_split
from utils.pytorchtools import EarlyStopping
from utils.data import load_IMDB_data
from utils.tools import evaluate_results_nc
from model import MAGNN_nc_AC

ap = argparse.ArgumentParser(description='MAGNN-AC testing for the IMDB dataset')
ap.add_argument('--layers', type=int, default=2, help='Number of layers. Default is 2.')
ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--save-postfix', default='IMDB', help='Postfix for the saved model and result. Default is DBLP.')
ap.add_argument('--feats-opt', type=str, default='111', help='010 means 1 type nodes use our processed feature')
ap.add_argument('--feats-drop-rate', type=float, default=0.3, help='The ratio of attributes to be dropped.')
ap.add_argument('--loss-lambda', type=float, default=0.5, help='Coefficient lambda to balance loss.')
ap.add_argument('--cuda', action='store_true', default=False, help='Using GPU or not.')
args = ap.parse_args()
print(args)

num_layers = args.layers
hidden_dim = args.hidden_dim
num_heads = args.num_heads
attn_vec_dim = args.attn_vec_dim
rnn_type = args.rnn_type
num_epochs = args.epoch
patience = args.patience
repeat = args.repeat
save_postfix = args.save_postfix
feats_drop_rate = args.feats_drop_rate
loss_lambda = args.loss_lambda
feats_opt = args.feats_opt
is_cuda = args.cuda

# random seed
seed = 234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if is_cuda:
    print('Using CUDA')
    torch.cuda.manual_seed(seed)

feats_opt = list(feats_opt)
feats_opt = list(map(int, feats_opt))
print('feats_opt: {}'.format(feats_opt))

# 0-MD 1-DM 2-MA 3-AM
etypes_lists = [[[0, 1], [2, 3]],
                [[1, 0], [1, 2, 3, 0]],
                [[3, 2], [3, 0, 1, 2]]]
num_metapaths = [2, 2, 2]
num_edge_type = 4
src_node_type = 0

# Params
out_dim = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
device = torch.device('cuda:0' if is_cuda else 'cpu')

# load data
nx_G_lists, edge_metapath_indices_lists, features_list, emb, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data()
features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
in_dims = [features.shape[1] for features in features_list]
emb_dim = emb.shape[1]
emb = torch.FloatTensor(emb).to(device)
adjM = torch.FloatTensor(adjM).to(device)
edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                               edge_metapath_indices_lists]
labels = torch.LongTensor(labels).to(device)
g_lists = []
for nx_G_list in nx_G_lists:
    g_lists.append([])
    for nx_G in nx_G_list:
        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(nx_G.number_of_nodes())
        g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
        g_lists[-1].append(g)
train_idx = train_val_test_idx['train_idx']
val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']

mask_list = []
for i in range(out_dim):
    mask_list.append(np.where(type_mask == i)[0])
for i in range(out_dim):
    mask_list[i] = torch.LongTensor(mask_list[i]).to(device)
feat_keep_idx, feat_drop_idx = train_test_split(np.arange(features_list[0].shape[0]), test_size=feats_drop_rate)
feat_keep_idx = torch.LongTensor(feat_keep_idx).to(device)
feat_drop_idx = torch.LongTensor(feat_drop_idx).to(device)
print('data load finish')

svm_macro_avg = np.zeros((7, ), dtype=np.float)
svm_micro_avg = np.zeros((7, ), dtype=np.float)
nmi_avg = 0
ari_avg = 0
print('start train with repeat = {}\n'.format(repeat))
for cur_repeat in range(repeat):
    print('cur_repeat = {}   ==============================================================='.format(cur_repeat))
    net = MAGNN_nc_AC(num_layers, num_metapaths, num_edge_type, etypes_lists, in_dims, emb_dim, hidden_dim,
                          out_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate, is_cuda, feats_opt)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    target_node_indices = np.where(type_mask == 0)[0]
    print('model init finish\n')

    # training loop
    print('training...')
    net.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
    for epoch in range(num_epochs):
        # training forward
        print('Epoch {}'.format(epoch))
        t = time.time()
        net.train()

        logits, _, _, loss_ac = net(
            (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
            (g_lists, type_mask, edge_metapath_indices_lists),
            target_node_indices)

        logp = F.log_softmax(logits, 1)
        loss_classification = F.nll_loss(logp[train_idx], labels[train_idx])
        train_loss = loss_classification + loss_lambda*loss_ac

        # autograd
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print('\ttrain_loss: {:.6f} | AC_loss: {:.6f} | classification_loss: {:.6f}'.format(
            train_loss.item(),
            loss_lambda*loss_ac.item(),
            loss_classification.item()))
        train_time = time.time() - t

        # validation forward
        t = time.time()
        net.eval()
        with torch.no_grad():
            logits, _, _, loss_ac = net(
                (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                (g_lists, type_mask, edge_metapath_indices_lists), target_node_indices)
            logp = F.log_softmax(logits, 1)
            loss_classification = F.nll_loss(logp[val_idx], labels[val_idx])
            val_loss = loss_classification + loss_lambda*loss_ac

            print('\tval_loss: {:.6f} | AC_loss: {:.6f} | classification_loss: {:.6f}'.format(
                val_loss.item(),
                loss_lambda*loss_ac.item(),
                loss_classification.item()))
        val_time = time.time() - t
        print('\ttrain time: {} | val time: {}'.format(train_time, val_time))

        early_stopping(loss_classification.item(), net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    # testing with evaluate_results_nc
    print('\ntesting...')
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
    net.eval()
    with torch.no_grad():
        _, embeddings, _, _ = net(
            (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
            (g_lists, type_mask, edge_metapath_indices_lists),
            target_node_indices)
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(
            embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)

        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari

svm_macro_avg = svm_macro_avg / repeat
svm_micro_avg = svm_micro_avg / repeat
nmi_avg /= repeat
ari_avg /= repeat
print('---\nThe average of {} results:'.format(repeat))
print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
print('NMI: {:.6f}'.format(nmi_avg))
print('ARI: {:.6f}'.format(ari_avg))
print('all finished')
