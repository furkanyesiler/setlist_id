import argparse
import itertools
import json
import os

import intervaltree
import numpy as np
import pandas as pd
from sklearn.svm import SVC

with open('concert_subsets.json') as f:
    CONCERT_SUBSETS = json.load(f)


def create_results(res_filename, gt_filename, skip=False, concerts=None,
                   is_print=True, return_analyzed=True):
    """
    Function to postprocess setlist identification results and analyze the
    final results

    :param res_filename: path to the results file
    :param gt_filename: path to the ground truth annotations
    :param skip: whether to skip every second result of the results
    :param concerts: list of concerts to consider
    :param is_print: whether to print computed evaluation metrics
    :param return_analyzed: whether to return the analyzed results. if false,
    it returns the postprocessed results without true/false annotations
    :return: postprocessed results
    """
    # reading results csv
    res = pd.read_csv(res_filename, header=None, index_col=None,
                      names=["concert", "q_start", "q_end", "metadata",
                             "r_start", "r_end", "conf"])

    # getting list of all concerts
    if concerts is None:
        concerts = res['concert'].drop_duplicates().tolist()
    else:
        concerts = CONCERT_SUBSETS[concerts]

    # obtaining postprocessed results per concert
    ind_res = []
    for concert in concerts:
        ind_res.append(postprocess_results(res, concert, skip))

    # merge results from individual concerts
    all_res = pd.concat(ind_res)

    all_res = all_res.astype(
        {"q_start": float, 'q_end': float, 'max_conf': float})
    all_res = all_res.sort_values(['concert', 'q_start']).reset_index(
        drop=True)

    gt_dict = create_gt_dict(gt_filename, concerts)
    results = analyze_matches(pd.DataFrame(all_res), gt_dict,
                              is_print=is_print)

    if return_analyzed:
        return results
    else:
        results = pd.DataFrame(all_res, columns=['concert', 'q_start', 'q_end',
                                                 'metadata', 'r_start',
                                                 'r_end', 'max_conf'])
        results = results.astype(
            {"q_start": float, 'q_end': float, 'max_conf': float})
        results = results.sort_values(['concert', 'q_start']).reset_index(
            drop=True)
        return results


def postprocess_results(df, concert, skip=False):
    """
    Function for postprocessing raw results. It performs 3 operations:
    (1) merging consecutive overlapping matches, (2) splitting all the overlaps
    and choosing the match with the highest confidence, and (3) merging
    consecutive overlapping matches.

    :param df: pandas dataframe to perform the operation
    :param concert: name of the concert
    :param skip: whether to skip every second result of the results
    :return: postprocessed results
    """
    # picking only one concert
    df = df[df['concert'] == concert]

    # whether to skip every second row
    if skip:
        df = df.iloc[::2, :].reset_index(drop=True)

    # merging consecutive overlapping matches
    df_res = collapse_rows(df)

    # obtain all possible overlaps
    df_o = split_overlapping_segments(df_res, df_res, 'q_start', 'q_end')

    # keeping the matches with highest confidence for overlapping segments
    df_o = df_o.astype(
        {"q_start": float, 'q_end': float, 'max_conf_res': float})
    df_o = df_o.sort_values(['_start_', 'max_conf_res'])
    df_o = df_o.drop_duplicates(subset=['_start_'], keep='last')

    # merging consecutive overlapping segments
    df_o = df_o[
        ['concert_res', '_start_', '_end_', 'metadata_res', 'r_start_res',
         'r_end_res', 'max_conf_res']].copy()
    df_o = df_o.rename(columns={'concert_res': 'concert', '_start_': 'q_start',
                                '_end_': 'q_end', 'metadata_res': 'metadata',
                                'r_start_res': 'r_start', 'r_end_res': 'r_end',
                                'max_conf_res': 'conf'})
    df_o = df_o.reset_index(drop=True)
    df_o = collapse_rows(df_o)

    return df_o


# merging consecutive overlapping matches
def collapse_rows(df):
    """
    Function for merging consecutive overlapping matches.

    :param df: pandas dataframe to perform merging
    """
    df = df.copy(deep=True)
    df['to_be_removed'] = False
    if 'conf' in df.columns:
        df['max_conf'] = df['conf']
    for i in df.index[1:]:
        if ((df.loc[i, 'metadata'] == df.loc[i - 1, 'metadata']) and
                (df.loc[i, 'q_start'] <= df.loc[i - 1, 'q_end'])):
            df.loc[i, 'q_start'] = df.loc[i - 1, 'q_start']
            df.loc[i - 1, 'to_be_removed'] = True
            df.loc[i, 'max_conf'] = max(df.loc[i - 1, 'max_conf'],
                                        df.loc[i, 'max_conf'])

    df['conf'] = df['max_conf']
    df = df[df['to_be_removed'] == False].drop(columns=['to_be_removed',
                                                        'conf']).reset_index(
        drop=True)
    return df


def split_overlapping_segments(report_df_gt, report_df_res,
                               start_col, end_col):
    """
    Function to split overlapping segments in a pandas dataframe

    :param report_df_gt:
    :param report_df_res:
    :param start_col:
    :param end_col:
    :return:
    """
    it = intervaltree.IntervalTree()
    for i in report_df_gt.index:
        it.addi(report_df_gt.loc[i, start_col],
                report_df_gt.loc[i, end_col],
                ('gt', i))
    for i in report_df_res.index:
        it.addi(report_df_res.loc[i, start_col],
                report_df_res.loc[i, end_col],
                ('res', i))
    it.split_overlaps()
    overlaps = {}
    for i in it:
        k = (i.begin, i.end)
        if k not in overlaps:
            overlaps[k] = {'gt': [],
                           'res': []}
        overlaps[k][i.data[0]].append(i.data[1])
    rows = []
    for k in overlaps:
        if len(overlaps[k]['gt']) == 0:
            overlaps[k]['gt'] = [None]
        if len(overlaps[k]['res']) == 0:
            overlaps[k]['res'] = [None]
        all_indices_combinations = itertools.product(overlaps[k]['gt'],
                                                     overlaps[k]['res'])
        for indices_combination in all_indices_combinations:
            gt_index, res_index = indices_combination
            start = k[0]
            end = k[1]
            rows.append([start,
                         end,
                         (end - start),
                         gt_index,
                         res_index])
    df = pd.DataFrame(rows, columns=['_start_',
                                     '_end_',
                                     '_length_',
                                     'gt_index',
                                     'res_index'])
    df = df.join(report_df_gt, on='gt_index', rsuffix='_gt')
    df = df.join(report_df_res, on='res_index', rsuffix='_res')
    df = df.sort_values(by=['_start_'])
    return df


def create_gt_dict(filename, concerts=None):
    """
    Function to create a dictionary for the ground truth annotations

    :param filename: path of the ground truth file
    :param concerts: whether to create the dictionary for a specific query
    :return: a dictionary that contains the ground truth annotations
    """
    gt = pd.read_csv(filename, header=None, index_col=None)
    gt_dict = {}
    for i in range(gt.iloc[:, 0].size):
        if concerts is None:
            query = gt.iloc[i, 0]
            ref = gt.iloc[i, 1]
            start = gt.iloc[i, 2]
            end = gt.iloc[i, 3]
            if query not in gt_dict:
                gt_dict[query] = {}
            if ref not in gt_dict[query]:
                gt_dict[query][ref] = []
            gt_dict[query][ref].append((start, end))
        else:
            if gt.iloc[i, 0] in concerts:
                query = gt.iloc[i, 0]
                ref = gt.iloc[i, 1]
                start = gt.iloc[i, 2]
                end = gt.iloc[i, 3]
                if query not in gt_dict:
                    gt_dict[query] = {}
                if ref not in gt_dict[query]:
                    gt_dict[query][ref] = []
                gt_dict[query][ref].append((start, end))
    return gt_dict


def analyze_matches(res, gt, is_print=False):
    """
    Function to analyze the results w.r.t. the ground truth and to print
    evaluation metrics

    :param res: results as a pandas dataframe
    :param gt: a dictionary that contains ground truth annotations
    :param is_print: whether to print the evaluation metrics
    :return: a pandas dataframe with the analyzed results
    """
    results = []
    tp = 0
    fn = 0
    fp = 0
    det_len_global = 0
    det_dict = {}
    det_dict_fn = {}
    for i in range(res.iloc[:, 0].size):
        status = 'fp'
        fp_doubt = 0
        query = res.iloc[i, 0]
        start = float(res.iloc[i, 1])
        end = float(res.iloc[i, 2])
        ref = res.iloc[i, 3]
        ref_start = res.iloc[i, 4]
        ref_end = res.iloc[i, 5]
        score = res.iloc[i, 6]
        det_len_tmp = 0
        if query in gt:  # whether the concert is in ground truth
            if ref in gt[query]:  # whether the reference is in the concert
                for item in gt[query][ref]:
                    if end >= item[0] and start <= item[1]:
                        det_len_tmp = min(end, item[1]) - max(start, item[0])
                        det_len_global += det_len_tmp
                        if ref in det_dict:
                            if det_dict[ref][1] > item[0] and \
                                    det_dict[ref][0] <= item[1]:
                                if end >= det_dict[ref][1] and start <= \
                                        det_dict[ref][0]:
                                    det_len_global -= min(end, det_dict[ref][
                                        1]) - max(start, det_dict[ref][0])
                                    det_dict[ref] = (
                                    min(det_dict[ref][0], max(start, item[0])),
                                    max(det_dict[ref][1], min(end, item[1])))
                        else:
                            det_dict[ref] = (max(start, item[0]),
                                             min(end, item[1]))
                        if ref not in det_dict_fn:
                            det_dict_fn[ref] = []
                        det_dict_fn[ref].append(item)
                        if status != 'tp':
                            tp += 1
                            status = 'tp'
                    else:
                        fp_doubt = 1
                if fp_doubt == 1 and status != 'tp':
                    det_len_tmp = 0
                    fp += 1

            else:
                fp += 1
                det_len_tmp = 0
            results.append((query, start, end, ref, ref_start, ref_end, score,
                            status, det_len_tmp))
        else:
            pass

    total_annotation_duration = 0
    total_annotations = 0

    for query in gt.keys():
        for ref in gt[query].keys():
            for item in gt[query][ref]:
                total_annotation_duration += item[1] - item[0]
                total_annotations += 1
            if ref not in det_dict:
                for item in gt[query][ref]:
                    results.append(
                        (query, item[0], item[1], ref, 0, 0, 0, 'fn', 0))
                    fn += 1
            else:
                for it in gt[query][ref]:
                    if it not in det_dict_fn[ref]:
                        results.append(
                            (query, it[0], it[1], ref, 0, 0, 0, 'fn', 0))
                        fn += 1
    if is_print:
        print('True positives: {}\n'
              'False positives: {}\n'
              'Total annotations: {}\n'
              'DAP : {:.3f}\n'
              'Detected_length hours: {:.2f} hours\n'
              'Total annotation length hours: {:.2f} hours\n'
              'Percentage of detected length: {:.3f}'.format(
               tp, fp, total_annotations,
               (total_annotations - fn) / total_annotations,
               det_len_global / 60 / 60, total_annotation_duration / 60 / 60,
               det_len_global / total_annotation_duration))

    columns = ['query', 'query_start', 'query_end', 'ref', 'ref_start',
               'ref_end', 'score', 'status', 'det_len']
    results = pd.DataFrame(results, columns=columns)
    results.sort_values(['query', 'query_start'], inplace=True)
    return results


def create_res_post_classification(res_val_filename, res_eval_filename,
                                   gt_val_filename, gt_eval_filename,
                                   skip=False, concerts=None):
    """
    The main function to create and analyze results. It analyzes the results
    for the validation data w.r.t. the ground truth for validation set. After
    obtaining correct and false matches on the validation data, it trains a
    support vector machine model for false positive filtering. Finally, it
    creates the results file for the evaluation data and applies the model.

    :param res_val_filename: path to the results file for the validation set
    :param res_eval_filename: path to the results file for the evaluation set
    :param gt_val_filename: path to the ground truth annotations for
    the validation set
    :param gt_eval_filename: path to the ground truth annotations for
    the evaluation set
    :param skip: whether to skip every second result on the res_eval file.
    :param concerts: which subset of concerts to create the results for.
    Default is all concerts
    """
    res_val = create_results(res_val_filename, gt_val_filename, skip=skip,
                             is_print=False)
    res_val = res_val[res_val['status'].isin(['fp', 'tp'])]
    res_val['status'].replace({'tp': 1, 'fp': 0}, inplace=True)
    data_training = np.array(list(
        zip(res_val['query_end'] - res_val['query_start'], res_val['score'],
            res_val['status']))).astype(np.float32)
    feat_train = data_training[:, :2]
    labels_train = data_training[:, -1]
    svc_model = SVC(class_weight='balanced')
    svc_model.fit(feat_train, labels_train)

    print('Postprocessed results before the classifier')

    res_eval = create_results(res_eval_filename, gt_eval_filename, skip=skip,
                              concerts=concerts, return_analyzed=False)

    data_test = np.array(list(zip(res_eval['q_end'] - res_eval['q_start'],
                                  res_eval['max_conf']))).astype(np.float32)

    feat_test = data_test[:, :2]
    pred_svc = svc_model.predict(feat_test)

    res_eval = res_eval[pred_svc == 1]
    res_eval.to_csv('res_eval_tmp.clean.csv', index=False, header=False)

    print('\nPostprocessed results after the classifier')

    create_results('res_eval_tmp.clean.csv', gt_eval_filename,
                   concerts=concerts)
    os.remove('res_eval_tmp.clean.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training and evaluation code for Re-MOVE experiments.')
    parser.add_argument('-rv',
                        '--res_val_filename',
                        type=str,
                        help='Path to the results file for '
                             'the validation set.')
    parser.add_argument('-re',
                        '--res_eval_filename',
                        type=str,
                        help='Path to the results file for '
                             'the evaluation set.')
    parser.add_argument('-gv',
                        '--gt_val_filename',
                        type=str,
                        help='Path to the ground truth annotations for '
                             'the validation set.')
    parser.add_argument('-ge',
                        '--gt_eval_filename',
                        type=str,
                        help='Path to the ground truth annotations for '
                             'the evaluation set.')
    parser.add_argument('--skip',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='Whether to skip every second result on '
                             'the res_eval file.')
    parser.add_argument('--concerts',
                        type=str,
                        default=None,
                        choices=['aq_a', 'aq_b', 'aq_c', 'pop', 'rock',
                                 'indie', 'rap', 'electronic'],
                        help='Which subset of concerts to create the results'
                             'for. Default is all concerts.')

    args = parser.parse_args()

    create_res_post_classification(args.res_val_filename,
                                   args.res_eval_filename,
                                   args.gt_val_filename,
                                   args.gt_eval_filename,
                                   args.skip,
                                   args.concerts)

