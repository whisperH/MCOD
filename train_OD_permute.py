import json
import os
import logging;

logging.basicConfig(level=logging.WARNING)
import numpy as np
import gc
import sys

sys.path.append("run_OD")
sys.path.append("/")

from DetectOutlier.dataset.label_data_generator import LabelDataGenerator
from DetectOutlier.model.PyOD import ModelFactory
from DetectOutlier.utils.utility import dict_combinations, metric, normalization
from DetectOutlier.utils.file_io import PathManager
from DetectOutlier.utils.parameters import get_hyper_para, setup, check_hyper_para
from visualization.visulize import visualize_tsne, visualize_KDE
import torch
import time
from copy import deepcopy
import pandas as pd
import pickle #pickle模块


# Perturbation defaults (edit here to enable; rounds > 1 turns on Monte-Carlo input UQ).
# `args.perturbation_*` overrides these when the CLI/config provides them.
PERTURBATION_ROUNDS = 10
PERTURBATION_SIGMA = 1.0
PERTURBATION_SEED = 46


# ---- ported from model_wrappers.py: Monte-Carlo input-perturbation uncertainty ----
def _primary_scores(raw_scores):
    """dict -> 'score' array (MCOD returns {"score": ...}); else ravel to 1-D."""
    if isinstance(raw_scores, dict):
        raw_scores = raw_scores.get("score")
    return np.asarray(raw_scores, dtype=float).reshape(-1)


def uncertainty_from_draws(scores, draws, sample_names):
    """Per-sample uncertainty from perturbation-round score draws.

    scores : baseline decision scores, shape (n,)
    draws  : scores across rounds, shape (B, n)
    Returns {sample_name: {score, score_mean, score_sd,
                           relative_uncertainty, relative_uncertainty_pct}}.
    """
    scores = _primary_scores(scores)
    draws = np.asarray(draws, dtype=float).reshape(len(draws), -1)
    out = {}
    for i, name in enumerate(sample_names):
        d = draws[:, i]
        mean = float(np.mean(d))
        sd = float(np.std(d, ddof=1))
        rel = sd / abs(mean) if mean != 0 else None
        out[str(name)] = {
            "score": float(scores[i]),
            "score_mean": mean,
            "score_sd": sd,
            "relative_uncertainty": float(rel) if rel is not None else None,
            "relative_uncertainty_pct": float(rel * 100) if rel is not None else None,
        }
    return out


class RunPipeline():
    def __init__(self, args, infer_flag=False):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        '''

        self.args = args
        self.save_model = args.save_model
        self.seed_list = list(np.arange(args.seed_nums) + 1)
        if infer_flag:
            self.logs_dir = args.infer_log_dir
            self.prepare_data_dir = args.prepare_infer_data_dir
            self.dataset_list = args.infer_datafile
        else:
            self.logs_dir = args.logs_dir
            self.prepare_data_dir = args.prepare_data_dir
            # data generator instantiation
            self.dataset_list = args.dataset_list_raw


    # results saving function
    def save_results(
            self, Y, y_pred, scores_pred,
            model_name, dataset_name,
            uncertainty_by_sample=None
    ):
        outlier_index = np.squeeze(np.argwhere(y_pred == 1))
        print(f"Model: {model_name} with \n {Y[outlier_index]}")
        if isinstance(scores_pred, dict):
            if 'score' in scores_pred:
                print(scores_pred["score"][outlier_index])

        all_index = [_ for _ in range(Y.shape[0])]
        list_outlier_index = np.squeeze(outlier_index).tolist()
        if type(list_outlier_index) == int:
            list_outlier_index = [list_outlier_index]

        normal_index = list(set(all_index) ^ set(list_outlier_index))
        outlier_id = Y[outlier_index]

        normal_id = Y[normal_index]

        para = dict()
        para['outlier_id'] = outlier_id
        para['normal_id'] = normal_id
        para['scores_pred'] = scores_pred
        para['all_sample_name'] = Y
        para['uncertainty_by_sample'] = uncertainty_by_sample

        assert len(normal_id) + len(outlier_id) == len(all_index), 'unmatched length with normal and outlier'
        if not PathManager.exists(os.path.join(self.logs_dir, "outliers")):
            PathManager.mkdirs(os.path.join(self.logs_dir, "outliers"))
        np.save(
            os.path.join(self.logs_dir, 'outliers', f'{dataset_name}.npy'),
            para
        )
        return normal_index, outlier_index

    def _perturbation_cfg(self):
        """Perturbation settings, mirroring model_wrappers.py config keys."""
        return (
            int(getattr(self.args, "perturbation_rounds", PERTURBATION_ROUNDS) or 0),
            float(getattr(self.args, "perturbation_sigma", PERTURBATION_SIGMA) or 0.08),
            int(getattr(self.args, "perturbation_seed", PERTURBATION_SEED) or 46),
        )

    def _perturbation_uncertainty(self, clf, X, scores_pred, sample_names):
        """Monte-Carlo input-perturbation uncertainty (from model_wrappers.py).

        Returns a {sample_name: metrics} dict, or None when disabled
        (perturbation_rounds <= 1).
        """
        rounds, sigma, seed = self._perturbation_cfg()
        if rounds <= 1:
            return None
        rng = np.random.default_rng(seed)
        draws = [
            _primary_scores(
                clf.decision_function(X + rng.normal(0.0, sigma, size=X.shape))
            )
            for _ in range(rounds)
        ]
        return uncertainty_from_draws(scores_pred, draws, sample_names)

    # run the experiment
    def runTrain(self, model_para):
        para_bak = model_para.copy()
        # for i, model_para in tqdm(enumerate(self.experiment_params)):
        model_name = model_para['model_name']

        dataset_name = model_para['dataset_list'].split(".csv")[0]
        prefix = f"seed_list-{model_para['seed_list']}-dataset_list-{dataset_name}"

        del model_para['model_name']
        del model_para['seed_list']
        del model_para['dataset_list']

        data_pair_file = os.path.join(self.prepare_data_dir, f"{prefix}.npy")
        data = np.load(data_pair_file, allow_pickle=True).item()

        X_train = data['X_train']

        sample_index_id = data['sample_index_id']
        y_train_label = data['y_train_label']
        X_test = data['X_test']
        y_test_label = data['y_test_label']

        sample_name = data['y_train_index']

        clf = ModelFactory(model_name, model_para)
        start_time = time.time()
        clf = clf.fit(
            X=X_train, y_train_label=y_train_label,
            sample_index_id=sample_index_id, sample_name=sample_name
        )
        end_time = time.time()

        time_fit = end_time - start_time
        print(f"fitting time is {time_fit}")
        X_data = X_test

        y_pred = clf.predict(X_data, return_confidence=False)
        scores_pred = clf.decision_scores_

        # Monte-Carlo input-perturbation uncertainty (from model_wrappers.py)
        uncertainty_by_sample = self._perturbation_uncertainty(
            clf, X_data, scores_pred, sample_name
        )

        if self.save_model:
            if not PathManager.exists(os.path.join(self.logs_dir, "model_weights")):
                PathManager.mkdirs(os.path.join(self.logs_dir, "model_weights"))
            if clf.method_type == "DeepLearning":
                torch.save(clf.model, os.path.join(self.logs_dir, "model_weights", f'{dataset_name}_model.pth'))
            with open(os.path.join(self.logs_dir, "model_weights", f'{dataset_name}_model.pickle'), 'wb') as f:
                pickle.dump(clf, f)

        self.save_results(
            sample_name, y_pred, scores_pred,
            model_name, dataset_name,
            uncertainty_by_sample=uncertainty_by_sample
        )

        del clf
        gc.collect()

    def runInfer(self, model_para, infer_data):
        model_name = model_para['model_name']
        sample_name = infer_data.index
        X = infer_data.values

        dataset_name = model_para['dataset_list'].split(".csv")[0]
        del model_para['model_name']
        del model_para['seed_list']
        del model_para['dataset_list']

        # clf = ModelFactory(model_name, model_para)
        with open(os.path.join(self.logs_dir, "model_weights", f'{dataset_name}_model.pickle'), 'rb') as f:
            clf = pickle.load(f)

        if clf.method_type == "DeepLearning":
            print("Trying to use the deep learning model")
            model_file = os.path.join(self.logs_dir, "model_weights", f'{dataset_name}_model.pth')
            clf.model = torch.load(model_file)
            # 推理必要的类字段
            clf.best_model_dict = clf.model.state_dict()
            if clf.preprocessing:
                clf.mean, clf.std = np.mean(X, axis=0), np.std(X, axis=0)

        # predict 里调用了 decision_function
        y_pred = clf.predict(X, return_confidence=False)

        # cacluating in function: decision_function
        scores_pred = clf.decision_scores_

        # Monte-Carlo input-perturbation uncertainty (from model_wrappers.py)
        uncertainty_by_sample = self._perturbation_uncertainty(
            clf, X, scores_pred, sample_name
        )
        print(uncertainty_by_sample)

        normal_index, outlier_index = self.save_results(
            sample_name, y_pred, scores_pred,
            model_name, dataset_name,
            uncertainty_by_sample=uncertainty_by_sample
        )
        del clf
        gc.collect()
        return normal_index, outlier_index, scores_pred

    def runEval(self, model_para, infer_data, model_name=None, **kwargs):
        # for i, model_para in tqdm(enumerate(self.experiment_params)):
        model_name = model_para.get('model_name', model_name)
        if "model_name" in model_para:
            del model_para['model_name']
        if "seed_list" in model_para:
            del model_para['seed_list']
        if "dataset_list" in model_para:
            del model_para['dataset_list']

        clf = ModelFactory(model_name, model_para)
        start_time = time.time()
        clf = clf.fit(
            X=infer_data['X_train'],
            **kwargs
        )
        end_time = time.time()
        time_fit = end_time - start_time
        print(f"fitting time is {time_fit}")

        infer_res = clf.decision_function(infer_data['X_test'])
        scores_pred = clf.decision_scores_
        # Monte-Carlo input-perturbation uncertainty (from model_wrappers.py)
        sample_name = infer_data['y_test_index']
        uncertainty_by_sample = self._perturbation_uncertainty(
            clf, infer_data['X_test'], scores_pred, sample_name
        )
        pd_uncertainty_by_sample = pd.DataFrame(uncertainty_by_sample)
        print(pd_uncertainty_by_sample.describe())
        os.makedirs(os.path.join(self.logs_dir, 'outliers'), exist_ok=True)
        saved_path = os.path.join(self.logs_dir, 'outliers', 'pd_uncertainty_by_sample.csv')
        pd_uncertainty_by_sample.T.to_csv(saved_path)
        print(f"uncertainty_by_sample are saved in {saved_path}")



        # performance
        y_true = infer_data['y_test_label']
        if type(infer_res) is np.ndarray:
            # pred_score = clf.decision_scores_
            result = metric(y_true=y_true, y_score=infer_res, pos_label=1)
            result['time_fit'] = time_fit
        elif type(infer_res) is dict:
            pred_score = infer_res['score']
            result = metric(y_true=y_true, y_score=pred_score, pos_label=1)
            result['time_fit'] = time_fit
        else:
            print(clf.decision_scores_)
            exit(-1)
        del clf
        gc.collect()
        return result, infer_res

def set_model_para(args, seed_list, dataset_list):
    # from pyod
    experiment_params = []
    for model_settings in args.models:
        combine_value_list = []
        combine_name_list = []

        model_name = list(model_settings.keys())[0]
        model_para = list(model_settings.values())[0]
        for ipara in model_para:
            combine_name_list.append(list(ipara.keys())[0])
            combine_value_list.append(list(ipara.values())[0])
        # combine_name_list.append('dist_list')
        # combine_value_list.append(args.dist_list)
        combine_name_list.append('seed_list')
        combine_value_list.append(seed_list)
        combine_name_list.append('dataset_list')
        combine_value_list.append(dataset_list)
        retList = dict_combinations(combine_value_list)
        for icombine in retList:
            temp_ = {"model_name": model_name}
            for para_name, para_value in zip(combine_name_list, icombine):
                temp_.update({para_name: para_value})
            experiment_params.append(temp_)
    return experiment_params

if __name__ == '__main__':
    seed = 46
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # run the above pipeline for reproducing the results in the paper

    command_line_args = get_hyper_para()
    print("Command Line Args:", command_line_args)
    args = setup(command_line_args)
    check_hyper_para(args)

    pipeline = RunPipeline(args)

    experiment_params = set_model_para(
        args, pipeline.seed_list,
        pipeline.dataset_list
    )

    if args.train_status:
        # S0.1 scale data and split into train and test set
        data_generator = LabelDataGenerator(args)
        scale_data, fill_mask = data_generator.preprocessor(args.data_dir, args.dataset_list_raw[0])
        data_generator.generator(
            dataset=scale_data,
            kf_num=args.seed_nums,
            saved_dir=args.prepare_data_dir,
            dataset_file=args.dataset_list_raw[0]
        )
        pipeline.runTrain(deepcopy(experiment_params[0]))
    elif args.infer_status:
        data_generator = LabelDataGenerator(args, infer_flag=True)
        infer_data, fill_mask = data_generator.preprocessor(
            data_dir=args.data_dir,
            dataset_file=args.infer_datafile
        )
        infer_dataname = args.infer_datafile.split(".csv")[0]
        normal_index, outlier_index, scores_pred = pipeline.runInfer(
            deepcopy(experiment_params[0]),
            infer_data
        )
        # load raw data for preparing clean data set
        raw_data = pd.read_csv(os.path.join(args.data_dir, args.infer_datafile), index_col=0)
        clean_data = raw_data.iloc[normal_index, :]

        clean_data.to_csv(
            os.path.join(args.infer_log_dir, f"{infer_dataname}_clean_result.csv")
        )

        outlier_data = raw_data.iloc[outlier_index.squeeze(), :]
        outlier_data.to_csv(
            os.path.join(args.infer_log_dir, f"{infer_dataname}_outlier_result.csv")
        )
    elif args.eval_status:
        from sklearn.metrics.pairwise import cosine_distances
        import seaborn as sns
        import matplotlib.pyplot as plt

        data_generator = LabelDataGenerator(args, infer_flag=True)
        eval_npz_data =  np.load(os.path.join(args.data_dir, args.infer_datafile.replace("gz", "npz")))
        feature_label = eval_npz_data.get("feature_label", None)
        sample_label = eval_npz_data['sample_label']
        eval_data = pd.read_csv(
            os.path.join(args.data_dir, args.infer_datafile.replace("npz", "gz")),
            index_col=0
        )
        infer_data, fill_mask = data_generator.preprocessor(
            raw_data = eval_data,
            data_dir=None,
            dataset_file=None
        )
        data_dict = data_generator.generator(
            infer_data, data_label=sample_label,
            fill_mask=fill_mask, TestonAll=args.TestonAll
        )

        result, infer_res = pipeline.runEval(
            deepcopy(experiment_params[0]),
            # infer_data.values[:, feature_label.values.astype(np.bool_)],
            data_dict,
            fill_mask=fill_mask,
            y_true=sample_label,
            feature_label=feature_label
        )
        print(result)

        json_str = json.dumps(result, indent=4)
        with open(os.path.join(args.infer_log_dir, "results.json"), 'w') as json_file:
            json_file.write(json_str)
            exit(0)

        # if feature_label is not None and 'feat_dist' in infer_res and args.visualize_stat:
        #     # print(f"feature abnormal counts: {}")
        #     Sort_count = np.argsort(-np.sum(feature_label, axis=0))
        #     feature_namelist = infer_data.columns.tolist()
        #     for i in range(len(Sort_count)):
        #         feat_data = np.concatenate(
        #             (
        #                 data_dict['X_train'][:, Sort_count[i]].reshape(-1, 1),
        #                 infer_res['feat_dist'][:, Sort_count[i]].reshape(-1, 1)
        #              ), axis=1
        #         )
        #         saved_filename = f"{experiment_params[0]['model_name']}"
        #         # if feature_namelist[Sort_count[i]] == "Q8TED0":
        #         if i < 100:
        #             visualize_KDE(
        #                 feat_data,
        #                 feature_namelist[Sort_count[i]],
        #                 feature_label[:, Sort_count[i]],
        #                 save_name=f"{os.path.join(args.infer_log_dir, saved_filename)}"
        #             )
                # if i == 100:
                #     break
            # saved_filename = f"{experiment_params[0]['model_name']}"
            # visualize_tsne(
            #     feat=infer_res['feat_dist'],
            #     sample_label=sample_label,
            #     save_name=f"{os.path.join(args.infer_log_dir, saved_filename)}"
            # )
#python train_OD_permute.py -c ../../configs/Hela/AE/AE_MEM10_S20.yaml data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile fake-shuffle_ratio-0.01-repeat_time-1.gz