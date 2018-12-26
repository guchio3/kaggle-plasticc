import numpy as np
import pandas as pd

from sklearn import preprocessing
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable


# これを更新する必要はありそう
class_weights_dict = {
    6: 18.8086925422,
    15: 18.2715315897,
    16: 18.8086925422,
    42: 18.8086925422,
    52: 18.8086925422,
    53: 18.809252663,
    62: 18.8086925422,
    64: 18.8086925422,
    65: 18.8086925422,
    67: 18.8086925422,
    88: 18.8086925422,
    90: 18.8086925422,
    92: 18.8086925422,
    95: 18.8086925422,
    #    99: 18.2712515266,
}

labeled_class_weights_dict = {
    0: 18.8086925422,
    1: 18.2715315897,
    2: 18.8086925422,
    3: 18.8086925422,
    4: 18.8086925422,
    5: 18.809252663,
    6: 18.8086925422,
    7: 18.8086925422,
    8: 18.8086925422,
    9: 18.8086925422,
    10: 18.8086925422,
    11: 18.8086925422,
    12: 18.8086925422,
    13: 18.8086925422,
    #    14: 18.2712515266,
}


lb = preprocessing.LabelBinarizer()
lb.fit(sorted(labeled_class_weights_dict.keys()))
class_weight_dict = labeled_class_weights_dict


def softmax(x, axis=1):
    z = np.exp(x)
    return z / np.sum(z, axis=axis, keepdims=True)


def weighted_multi_logloss(y_true, y_pred):
    '''
    ↓ のような input を期待
    [
        [0.1, 0.3, 0.6, 0.0, 0.0, ...],
        [0.0, 0.0, 0.8, 0.1, 0.0, ...],
        [0.1, 0.0, 0.2, 0.0, 0.0, ...],
    ]

    '''
    y_pred = np.clip(y_pred, 10**(-15), 1 - 10**(-15))
    y_pred = np.reshape(y_pred, (-1, 14))
    weights = np.array([class_weight_dict[key]
                        for key in sorted(class_weight_dict.keys())])
    num_classes = [np.sum(y_true == key)
                   for key in sorted(class_weight_dict.keys())]
    true_mask = lb.transform(y_true)
    score = -np.sum((weights / num_classes) * true_mask *
                      np.log(y_pred)) / np.sum(weights)
    return 'wloss', score, False


def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    #classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    #class_weight = labeled_class_weights_dict
    if len(np.unique(y_true)) > 14:
        classes.append(14)
        # classes.append(99)
        class_weight[14] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k]
                          for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


def wloss_metric(preds, train_data):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    weight_tensor = torch.tensor(list(class_weight.values()),
                             requires_grad=False).type(torch.FloatTensor)
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    y_h /= y_h.sum(dim=0, keepdim=True)
    y_p = torch.tensor(preds, requires_grad=False).type(torch.FloatTensor)
    if len(y_p.shape) == 1:
        y_p = y_p.reshape(len(classes), -1).transpose(0, 1)
    ln_p = torch.log_softmax(y_p, dim=1)
    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll) / torch.sum(weight_tensor)
    return 'wloss', loss.numpy() * 1., False


def wloss_objective(preds, train_data):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    #class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 2, 53: 1, 62: 1, 64: 2, 65: 1, 67: 2, 88: 1, 90: 1, 92: 1, 95: 1}
    weight_tensor = torch.tensor(list(class_weight.values()),
                             requires_grad=False).type(torch.FloatTensor)
    class_dict = {c: i for i, c in enumerate(classes)}
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    ys = y_h.sum(dim=0, keepdim=True)
    y_h /= ys
    y_p = torch.tensor(preds, requires_grad=True).type(torch.FloatTensor)
    y_r = y_p.reshape(len(classes), -1).transpose(0, 1)
    ln_p = torch.log_softmax(y_r, dim=1)
    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll)
    grads = grad(loss, y_p, create_graph=True)[0]
    grads *= float(len(classes)) / torch.sum(1 / ys)  # scale up grads
    hess = torch.ones(y_p.shape)  # haven't bothered with properly doing hessian yet
    return grads.detach().numpy(), \
        hess.detach().numpy()


def calc_team_score(y_true, y_preds):
    '''
    y_true:１次元のnp.array
    y_pred:softmax後の１4次元のnp.array
    '''
    class99_prob = 1/9
    class99_weight = 2
            
    y_p = y_preds * (1-class99_prob)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    
    y_true_ohe = pd.get_dummies(y_true).values
    nb_pos = y_true_ohe.sum(axis=0).astype(float)
    
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    
    y_log_ones = np.sum(y_true_ohe * y_p_log, axis=0)
    y_w = y_log_ones * class_arr / nb_pos
    score = - np.sum(y_w) / (np.sum(class_arr)+class99_weight)\
        + (class99_weight/(np.sum(class_arr)+class99_weight))*(-np.log(class99_prob))

    return score


def wloss_metric_for_zeropad(preds, train_data, gal_cols, ext_gal_cols, gal_rows, ext_gal_rows):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    weight_tensor = torch.tensor(list(class_weight.values()),
                             requires_grad=False).type(torch.FloatTensor)
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    y_h /= y_h.sum(dim=0, keepdim=True)
    y_p = torch.tensor(preds, requires_grad=False).type(torch.FloatTensor)
    if len(y_p.shape) == 1:
        y_p = y_p.reshape(len(classes), -1).transpose(0, 1)
    p = pd.DataFrame(torch.softmax(y_p, dim=0).numpy())
    p.loc[ext_gal_rows, gal_cols] = 0.
    p.loc[gal_rows, ext_gal_cols] = 0.
    p = np.clip(a=p.values/np.sum(p.values, axis=1).reshape((-1, 1)), a_min=1e-15, a_max=1 - 1e-15)
    ln_p = np.log(p)
    ln_p = torch.tensor(ln_p, requires_grad=False).type(torch.FloatTensor)
#    ln_p = torch.log_softmax(y_p, dim=1)
    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll) / torch.sum(weight_tensor)
    return loss.numpy() * 1.


def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))

def _gumbel_softmax_sample(logits, tau=0.1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = Variable(_sample_gumbel(logits.size(), eps=eps, out=logits.data.new()))
    y = logits + gumbel_noise
    return F.softmax(y / tau, dims - 1)


def wloss_objective_gumbel(preds, train_data):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    #class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 2, 53: 1, 62: 1, 64: 2, 65: 1, 67: 2, 88: 1, 90: 1, 92: 1, 95: 1}
    weight_tensor = torch.tensor(list(class_weight.values()),
                             requires_grad=False).type(torch.FloatTensor)
    class_dict = {c: i for i, c in enumerate(classes)}
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    ys = y_h.sum(dim=0, keepdim=True)
    y_h /= ys
    y_p = torch.tensor(preds, requires_grad=True).type(torch.FloatTensor)
    y_r = y_p.reshape(len(classes), -1).transpose(0, 1)
    y_r = torch.clamp(y_r, e-15, 1-e-15)
    ln_p = _gumbel_softmax_sample(torch.log(y_r))
#    ln_p = torch.log_softmax(y_r, dim=1)
    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll)
    grads = grad(loss, y_p, create_graph=True)[0]
    grads *= float(len(classes)) / torch.sum(1 / ys)  # scale up grads
    hess = torch.ones(y_p.shape)  # haven't bothered with properly doing hessian yet
    return grads.detach().numpy(), \
        hess.detach().numpy()


def wloss_metric_gumbel(preds, train_data):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    weight_tensor = torch.tensor(list(class_weight.values()),
                             requires_grad=False).type(torch.FloatTensor)
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    y_h /= y_h.sum(dim=0, keepdim=True)
    y_p = torch.tensor(preds, requires_grad=False).type(torch.FloatTensor)
    if len(y_p.shape) == 1:
        y_p = y_p.reshape(len(classes), -1).transpose(0, 1)
    #ln_p = torch.log_softmax(y_p, dim=1)
    y_p = torch.clamp(y_p, e-15, 1-e-15)
    ln_p = _gumbel_softmax_sample(torch.log(y_p))
    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll) / torch.sum(weight_tensor)
    return 'wloss', loss.numpy() * 1., False



