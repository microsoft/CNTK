import pickle, sys
import numpy as np
from collections import defaultdict
from scipy import sparse as sp


def load_batch_data(file_name, params):
    batch_size = params['batch_size']
    labels = []
    fm_features = []
    dnn_feats_index = []
    dnn_feats_val = []
    cnt = 0
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                break
            cols = line.strip().split(' ')
            label = float(cols[0])
            if label > 0:
                label = [1]
            else:
                label = [0]
            cur_fm_features = []
            feat_index_res = defaultdict(lambda: [])
            feat_val_res = defaultdict(lambda: [])
            for word in cols[1:]:
                tokens = word.split(':')
                cur_fm_features.append([int(tokens[1]) - 1, float(tokens[2])])
                feat_index_res[int(tokens[0]) - 1].append(int(tokens[1]) - 1)
                feat_val_res[int(tokens[0]) - 1].append(float(tokens[2]))
            fm_features.append(cur_fm_features)
            labels.append(label)
            sorted_feat_index_res = sorted(feat_index_res.items(), key=lambda x: x[0])
            sorted_feat_val_res = sorted(feat_val_res.items(), key=lambda x: x[0])
            feat_index = [item[1] for item in sorted_feat_index_res]
            feat_val = [item[1] for item in sorted_feat_val_res]
            dnn_feats_index.append(feat_index)
            dnn_feats_val.append(feat_val)
            cnt += 1
            if cnt == batch_size:
                yield labels, fm_features, dnn_feats_index, dnn_feats_val
                cnt = 0
                labels = []
                fm_features = []
                dnn_feats_index = []
                dnn_feats_val = []
        # if cnt > 0:
        #    yield labels, fm_features, dnn_feats_index, dnn_feats_val


def convertToSparseMatrix(labels, fm_features, dnn_feats_index, dnn_feats_val, params):
    feature_dim = params['feature_cnt']
    field_cnt = params['field_cnt']
    # convert fm feature
    instance_cnt = len(fm_features)
    fm_rows = []
    fm_cols = []
    fm_vals = []
    for i in range(instance_cnt):
        for feats in fm_features[i]:
            fm_rows.append(i)
            fm_cols.append(feats[0])
            fm_vals.append(feats[1])
    row = np.array(fm_rows, dtype=np.int64)
    col = np.array(fm_cols, dtype=np.int64)
    data = np.array(fm_vals, dtype=np.float32)
    lr_input_sp = sp.csr_matrix((data, (row, col)), shape=(instance_cnt, feature_dim))
    fm_vals_square = [val * val for val in fm_vals]
    data_square = np.array(fm_vals_square, dtype=np.float32)
    fm_input_sp = sp.csr_matrix((data_square, (row, col)), shape=(instance_cnt, feature_dim))

    # convert dnn features
    dnn_rows = []
    dnn_cols = []
    dnn_vals = []
    instance_cnt = len(dnn_feats_index)
    for i in range(instance_cnt):
        for j in range(len(dnn_feats_index[i])):
            for k in range(len(dnn_feats_index[i][j])):
                dnn_rows.append(i)
                dnn_cols.append(j * feature_dim + dnn_feats_index[i][j][k])
                dnn_vals.append(dnn_feats_val[i][j][k])
    dnn_row = np.array(dnn_rows, dtype=np.int64)
    dnn_col = np.array(dnn_cols, dtype=np.int64)
    dnn_val = np.array(dnn_vals, dtype=np.float32)
    dnn_input_sp = sp.csr_matrix((dnn_val, (dnn_row, dnn_col)), shape=(instance_cnt, field_cnt * feature_dim))
    res = {}
    res['lr_input'] = lr_input_sp
    res['fm_input'] = fm_input_sp
    res['dnn_input'] = dnn_input_sp
    res['labels'] = np.asarray(labels, dtype=np.float32)
    return res


def pre_build_data_cache(infile, outfile, params):
    wt = open(outfile, 'wb')
    for labels, fm_features, dnn_feats_index, dnn_feats_val in load_batch_data(infile, params):
        input_in_sp = convertToSparseMatrix(labels, fm_features, dnn_feats_index, dnn_feats_val, params)
        pickle.dump((input_in_sp), wt)
    wt.close()


def load_data_cache(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
