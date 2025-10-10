# -*- coding: utf-8 -*-
"""Using AutoEncoder with Outlier Detection (PyTorch)
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import collections
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn

from .base import BaseDetector
from DetectOutlier.utils.stat_models import pairwise_distances_no_broadcast
from DetectOutlier.utils.torch_utility import get_activation_by_name
from DetectOutlier.model.loss.MemoryLoss import ClusterMemory

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None, pseudo_y=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.pseudo_y = pseudo_y
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)
        if self.pseudo_y is not None:
            pseudo_label = self.pseudo_y[idx, :]
            return torch.from_numpy(sample), idx, pseudo_label
        else:
            return torch.from_numpy(sample), idx

class AR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=None, activation=None):
        super(AR_Block, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.bn_layer = torch.nn.BatchNorm1d(out_channels)
        self.activation = activation
        if dropout_rate is not None:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.linear(x)
        out = self.bn_layer(out)
        if self.activation is not None:
            out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class InnerResidualAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=(128, 64),
                 dropout_rate=0.2,
                 latent_dim=2,
                 batch_norm=True,
                 hidden_activation='relu'):

        # initialize the super class
        super(InnerResidualAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation

        # create the dimensions for the input and hidden layers
        self.layers_neurons_encoder_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_encoder_[::-1]

        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        self.encoder0 = AR_Block(
            self.layers_neurons_encoder_[0], self.layers_neurons_encoder_[1],
            dropout_rate, activation=self.activation
        )
        self.encoder1 = AR_Block(
            self.layers_neurons_encoder_[1], self.layers_neurons_encoder_[2],
            dropout_rate, activation=self.activation
        )
        self.encoder2 = AR_Block(
            self.layers_neurons_encoder_[2], self.layers_neurons_encoder_[3],
            dropout_rate, activation=self.activation
        )
        self.encoder3 = AR_Block(
            self.layers_neurons_encoder_[3], self.layers_neurons_encoder_[4],
            dropout_rate, activation=self.activation
        )
        self.mu_layer = torch.nn.Linear(
            self.layers_neurons_encoder_[-1],
            latent_dim
        )
        self.sigma_layer = torch.nn.Linear(
            self.layers_neurons_encoder_[-1],
            latent_dim
        )
        self.layers_neurons_decoder_ = [latent_dim] + self.layers_neurons_decoder_
        self.decoder0 = AR_Block(
            self.layers_neurons_decoder_[0], self.layers_neurons_decoder_[1],
            dropout_rate, activation=self.activation
        )
        self.decoder1 = AR_Block(
            self.layers_neurons_decoder_[1], self.layers_neurons_decoder_[2],
            dropout_rate, activation=self.activation
        )
        self.decoder2 = AR_Block(
            self.layers_neurons_decoder_[2], self.layers_neurons_decoder_[3],
            dropout_rate, activation=self.activation
        )
        self.decoder3 = AR_Block(
            self.layers_neurons_decoder_[2], self.layers_neurons_decoder_[3],
            dropout_rate=None, activation=None
        )


        # set memory module:
        self.memory = None
    def forward(self, x):
        # we could return the latent representation here after the encoder
        # as the latent representation
        out = self.encoder0(x)
        out = self.encoder1(out)
        out = self.encoder2(out)
        mid_x_feat = self.encoder3(out)

        mu = self.mu_layer(mid_x_feat)
        std = torch.exp(0.5 * self.sigma_layer(mid_x_feat))
        eps = torch.randn_like(std).to(std.device)
        z = mu + eps * std

        out = self.decoder1(z)
        out = self.decoder2(out)
        x_feat = self.decoder3(out)
        return {
            "laten_vec":mid_x_feat,
            "latent_h": z,
            "X_hat": x_feat,
            "mu": mu,
            "sigma": std,
        }


class AdaptiveResidualAE(BaseDetector):
    """Auto Encoder (AE) is a type of neural networks for learning useful data
    representations in an unsupervised manner. Similar to PCA, AE could be used
    to detect outlying objects in the data by calculating the reconstruction
    errors. See :cite:`aggarwal2015outlier` Chapter 3 for details.

    Notes
    -----
        This is the PyTorch version of AutoEncoder. See auto_encoder.py for
        the TensorFlow version.

        The documentation is not finished!

    Parameters
    ----------
    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. So the network has the
        structure as [n_features, 64, 32, 32, 64, n_features]

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://pytorch.org/docs/stable/nn.html for details.
        Currently only
        'relu': nn.ReLU()
        'sigmoid': nn.Sigmoid()
        'tanh': nn.Tanh()
        are supported. See pyod/utils/torch_utility.py for details.

    batch_norm : boolean, optional (default=True)
        Whether to apply Batch Normalization,
        See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

    learning_rate : float, optional (default=1e-3)
        Learning rate for the optimizer. This learning_rate is given to
        an Adam optimizer (torch.optim.Adam).
        See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    weight_decay : float, optional (default=1e-5)
        The weight decay for Adam optimizer.
        See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    loss_fn : obj, optional (default=torch.nn.MSELoss)
        Optimizer instance which implements torch.nn._Loss.
        One of https://pytorch.org/docs/stable/nn.html#loss-functions
        or a custom loss. Custom losses are currently unstable.

    verbose : int, optional (default=1)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.
        !CURRENTLY NOT SUPPORTED.!

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.
        !CURRENTLY NOT SUPPORTED.!

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : Keras Object
        The underlying AutoEncoder in Keras.

    history_: Keras Object
        The AutoEncoder training history.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 batch_norm=True,
                 learning_rate=1e-3,
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 weight_decay=1e-5,
                 # validation_size=0.1,
                 preprocessing=True,
                 loss_fn_name="MSE",
                 start_pesudo_epoch=80,
                 # verbose=1,
                 # random_state=None,
                 contamination=0.1,
                 mem_temp=0.02,
                 use_hard=False,
                 mem_momentum=0.05,
                 mem_loss_weight=0.05,
                 kl_loss_weight=0.05,
                 device=None):
        super(AdaptiveResidualAE, self).__init__(contamination=contamination)

        self.method_type = "DeepLearning"

        # save the initialization values
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.loss_fn_name = loss_fn_name
        self.start_pesudo_epoch = start_pesudo_epoch
        # self.verbose = verbose
        self.device = device
        self.mem_temp = mem_temp
        self.use_hard = use_hard
        self.mem_momentum = mem_momentum
        self.mem_loss_weight = mem_loss_weight
        self.kl_loss_weight = kl_loss_weight

        # # create default loss functions
        if "MSE" in self.loss_fn_name:
            self.loss_fn = torch.nn.MSELoss()

        # create default calculation device (support GPU if available)
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # default values for the amount of hidden neurons
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None, **kwargs):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        self.n_samples, self.n_features = X.shape[0], X.shape[1]

        # initialize the model
        self.model = InnerResidualAutoencoder(
            n_features=self.n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation)

        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(X)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)

        self._process_decision_scores()
        return self

    @torch.no_grad()
    def generate_cluster_features(self, labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).squeeze()
        return centers

    def _train_autoencoder(self, X):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        # conduct standardization if needed
        pseudo_y = np.expand_dims(np.array([-100]*X.shape[0]), axis=1)
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std, pseudo_y=pseudo_y)
            test_set = PyODDataset(X=X, mean=self.mean, std=self.std, pseudo_y=pseudo_y)

        else:
            train_set = PyODDataset(X=X, pseudo_y=pseudo_y)
            test_set = PyODDataset(X=X, pseudo_y=pseudo_y)

        train_loader = IterLoader(torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True))

        # shuffle must be false, assure correct sample index of each epoch
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

        self.best_loss = float('inf')
        self.best_ctr_acc = 0.0
        self.best_acc = 0.0
        self.best_model_dict = None
        history_outlier_info = dict()
        history_outlier_label = np.zeros([self.n_samples, ])

        for epoch in range(self.epochs):
            train_loader.new_epoch()
            self.model.train()
            overall_loss = []
            # for data, data_idx, batch_y_pseudo_label in train_loader:
            for i in range(len(train_loader)):
                data, data_idx, batch_y_pseudo_label = train_loader.next()

                data = data.to(self.device).float()
                model_output = self.model(data)
                if "MSE" in self.loss_fn_name:
                    loss = self.loss_fn(data, model_output['X_hat'])
                    # kl loss
                    kl_loss = (model_output['sigma'] ** 2 + model_output['mu'] ** 2 - torch.log(
                        model_output['sigma']) - 1 / 2).sum()
                    loss += self.kl_loss_weight * kl_loss.item()

                if ("mem_loss" in self.loss_fn_name) and (epoch > self.start_pesudo_epoch):
                    batch_y_pseudo_label = batch_y_pseudo_label.to(self.device)
                    memory_res = self.model.memory(
                        model_output['latent_h'],
                        batch_y_pseudo_label.squeeze().long()
                    )
                    loss += self.mem_loss_weight * memory_res['CE_loss'].item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())

            print('epoch {epoch}: training loss {train_loss} '.format(
                epoch=epoch, train_loss=np.mean(overall_loss)))

            print("===========> collect pseudo label <==================")
            if "mem_loss" in self.loss_fn_name:
                # collect history outlier data
                self.model.eval()
                with torch.no_grad():
                    features = []
                    outlier_scores = np.zeros([self.n_samples, ])
                    for data, data_idx, _ in test_loader:
                        data_cuda = data.to(self.device).float()
                        infer_res = self.model(data_cuda)
                        features += infer_res['latent_h'].split(1)
                        outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                            data.cpu().numpy(),
                            infer_res['X_hat'].cpu().numpy()
                        )
                    pseudo_threshold = np.percentile(
                        outlier_scores,
                        100 * (1 - self.contamination)
                    )
                    pseudo_labels = (outlier_scores > pseudo_threshold).astype('int').ravel()
                    history_outlier_label += pseudo_labels

                    for data_idx, data_score in enumerate(outlier_scores.ravel()):
                        if data_score > pseudo_threshold:
                            if data_idx not in history_outlier_info:
                                history_outlier_info[data_idx] = {
                                    "num": 0,
                                    "score_list": [],
                                }
                            history_outlier_info[data_idx]["num"] += 1
                            history_outlier_info[data_idx]["score_list"].append(data_score)
                    if epoch == self.start_pesudo_epoch:
                        # create memory cluster
                        pseudo_inliner_list = []
                        pseudo_inliner_features = []
                        for _, data_idx, _ in test_loader:
                            for idx in data_idx.cpu().numpy().tolist():
                                if idx not in history_outlier_info:
                                    pseudo_inliner_list.append(0)
                                else:
                                    pseudo_inliner_list.append(1)
                                pseudo_inliner_features.append(features[idx])
                        # generate new dataset and calculate cluster centers
                        cluster_features = self.generate_cluster_features(pseudo_inliner_list,
                                                                          pseudo_inliner_features)

                        # Create hybrid memory
                        self.model.memory = ClusterMemory(
                            self.hidden_neurons[-1], num_cluster=2,
                            temp=self.mem_temp, momentum=self.mem_momentum, use_hard=self.use_hard
                        ).cuda()
                        self.model.memory.features = torch.nn.functional.normalize(cluster_features, dim=1).cuda()

                        self.best_loss = float('inf')
                        self.best_ctr_acc = 0.0
                        self.best_acc = 0.0
                    if epoch >= self.start_pesudo_epoch:
                        # collect new dataset
                        history_outlier_label[history_outlier_label > 0] = 1
                        pseudo_labeled_dataset = PyODDataset(
                            X=X, mean=self.mean, std=self.std,
                            pseudo_y=np.expand_dims((pseudo_labels), axis=1)
                        )
                        train_loader = IterLoader(torch.utils.data.DataLoader(pseudo_labeled_dataset,
                                                                              batch_size=self.batch_size,
                                                                              shuffle=True))





            # track the best model so far
            if np.mean(overall_loss) <= self.best_loss:
                # print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = self.model.state_dict()

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model', 'best_model_dict'])
        X = check_array(X)

        # note the shuffle may be true but should be False
        if self.preprocessing:
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        outlier_mem_scores = np.zeros([X.shape[0], 2])
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                # this is the outlier score
                model_output = self.model(data_cuda)
                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    data, model_output['X_hat'].cpu().numpy())
                outlier_mem_scores[data_idx, :] = self.model.memory(
                    model_output['latent_h']
                )['logits'].cpu().numpy()

        return {
            'score': outlier_scores,
            'memo_score': outlier_mem_scores,
        }


    def _process_decision_scores(self, sample_name=None):
        """Internal function to calculate key attributes:

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """
        # np.where(sample_name.values=='iBAQ 38')[0]
        self.threshold_ = np.percentile(self.decision_scores_['score'],
                                     100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_['score'] > self.threshold_).astype(
            'int').ravel()

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_['score'])
        self._sigma = np.std(self.decision_scores_['score'])

        return self

    def predict(self, X, return_confidence=False, **kwargs):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        return_confidence : boolean, optional(default=False)
            If True, also return the confidence of prediction.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        confidence : numpy array of shape (n_samples,).
            Only if return_confidence is set to True.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        pred_score = self.decision_function(X)
        prediction_AE = (pred_score['score'] > self.threshold_).astype('int').ravel()
        prediction_memo = pred_score['memo_score'].argmax(axis=1)

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction_memo, confidence

        return prediction_AE+prediction_memo