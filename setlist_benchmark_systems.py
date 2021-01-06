import argparse
import math
import os
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf  # noqa E402
from essentia.standard import ChromaCrossSimilarity, CoverSongSimilarity
from joblib import Parallel, delayed

from utils import create_re_move_model
from utils import extract_2dftm
from utils import pairwise_cosine_distance
from utils import pairwise_euclidean_distance

# for limiting keras usage to N threads
if 'NUM_THREADS' in os.environ:
    NUM_THREADS = os.environ['NUM_THREADS']
    tf.config.threading.set_intra_op_parallelism_threads(int(NUM_THREADS))
    tf.config.threading.set_inter_op_parallelism_threads(int(NUM_THREADS))

# for ignoring tensorflow info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore", category=DeprecationWarning)


def compute_distance_qmax(query_feat, ref_feat):
    """
    Compute the Qmax distance between two songs using Essentia.

    :param query_feat: chroma feature for the query
    :param ref_feat: chroma feature for the reference
    :return: Qmax distance between the query and the reference
    """
    csm_ess = ChromaCrossSimilarity()
    css_ess = CoverSongSimilarity()
    csm = csm_ess(query_feat.T, ref_feat.T)
    _, distance = css_ess(csm)
    return distance


def prepare_output_mul_queries(query_paths, query_names, references,
                               reference_names, output, system, sample_rate,
                               segment_size, hop_size, num_workers, ref_sizes):
    """
    Function to process multiple queries (concerts) one by one.

    :param query_paths: paths to the chroma feature of the query concert
    :param query_names: names of the queries
    :param references: embeddings or chroma features of the references
    :param reference_names: names of the references
    :param output: path to the output file to save
    :param system: name of the system to use
    :param sample_rate: sample rate of the audio files
    :param segment_size: size of the analysis windows in seconds
    :param hop_size: hop size between analysis windows in seconds
    :param num_workers: number of workers for parallel processing
    :param ref_sizes: size of the reference songs in frames
    """
    all_res = []
    for i in range(len(query_paths)):
        query_tmp = np.load(query_paths[i])
        res_tmp = prepare_output_single_query(query_tmp, query_names[i],
                                              references, reference_names,
                                              output, system, sample_rate,
                                              segment_size, hop_size,
                                              num_workers, ref_sizes, True)
        all_res.extend(res_tmp)

    pd.DataFrame(all_res).to_csv(output, header=None, index=None)


def prepare_output_single_query(query_feat, query_name, references,
                                reference_names, output, system, sample_rate,
                                segment_size, hop_size, num_workers,
                                ref_sizes, mul_queries):
    """
    Function to process one query (concert) against all the references.

    :param query_feat: chroma feature of the query concert
    :param query_name: names of the query concert
    :param references: embeddings or chroma features of the references
    :param reference_names: names of the references
    :param output: path to the output file to save
    :param system: name of the system to use
    :param sample_rate: sample rate of the audio files
    :param segment_size: size of the analysis windows in seconds
    :param hop_size: hop size between analysis windows in seconds
    :param num_workers: number of workers for parallel processing
    :param ref_sizes: size of the reference songs in frames
    :param mul_queries: whether to save the results or to return them
    """
    # getting the number of frames of the query
    query_len = query_feat.shape[1]

    # computing the ideal hop size in samples
    # the goal is to keep the temporal resolution used by the crema model
    hop_in_samples = int(sample_rate * 4096 / 44100)

    # computing the segment size and hop size in frames
    segment_in_frames = int(segment_size * sample_rate / hop_in_samples)
    hop_in_frames = int(hop_size * sample_rate / hop_in_samples)

    # computing the number of segments in the query
    num_segments = math.ceil(
        (query_len - segment_in_frames) / hop_in_frames) + 1

    # initializing the score matrix
    scores = np.zeros((num_segments, len(references)), dtype=np.float32)

    # dividing the full query into segments for parallel processing
    tmp_query_list = []
    for i in range(num_segments):
        tmp_query = query_feat[:,
                               hop_in_frames * i:
                               hop_in_frames * i + segment_in_frames]
        tmp_query_list.append(tmp_query)

    # loading the onset strength if 2dftm is used
    if system == '2dftm':
        onset_file = './data/ASID_onsets/{}_onset.npy'.format(query_name)
        onset_env = np.load(onset_file)
        # the onset strength is computed using sample rate 22050 and
        # hop size 512
        hop_onset = int(hop_size * 22050 / 512)
        segment_onset = int(segment_size * 22050 / 512)

    # if parallel processing is used
    if num_workers > 0:
        # computing re-move embeddings for the query segments
        if system == 're-move':
            proc_query = Parallel(n_jobs=num_workers,
                                  backend='multiprocessing')(
                delayed(get_re_move_emb)(
                    tmp_query_list[i],
                    query_name)
                for i in range(len(tmp_query_list)))
            tmp_query_list = proc_query
        # computing 2dftm embeddings for the query segments
        if system == '2dftm':
            proc_query = Parallel(n_jobs=num_workers,
                                  backend='multiprocessing')(
                delayed(get_2dftm_query)(
                    tmp_query_list[i],
                    onset_env[hop_onset * i:hop_onset * i + segment_onset])
                for i in range(len(tmp_query_list)))
            tmp_query_list = proc_query
    # if parallel processing is not used
    else:
        # computing re-move embeddings for the query segments
        if system == 're-move':
            proc_query = [get_re_move_emb(
                tmp_query_list[i],
                query_name)
                for i in range(len(tmp_query_list))]
            tmp_query_list = proc_query
        # computing 2dftm embeddings for the query segments
        if system == '2dftm':
            proc_query = [get_2dftm_query(
                tmp_query_list[i],
                onset_env[hop_onset * i:hop_onset * i + segment_onset])
                for i in range(len(tmp_query_list))]
            tmp_query_list = proc_query

    # if re-move is used, compute cosine distance in a vectorized way
    if system == 're-move':
        scores = pairwise_cosine_distance(
            np.concatenate(tmp_query_list, axis=0),
            np.concatenate(references, axis=0))
    # if 2dftm is used, compute euclidean distance in a vectorized way
    elif system == '2dftm':
        scores = pairwise_euclidean_distance(
            np.concatenate(tmp_query_list, axis=0),
            np.concatenate(references, axis=0))
    else:
        # putting pairs of (query segment, reference) into a list for parallel
        # processing
        emb_pairs = []
        for i in range(len(tmp_query_list)):
            for j in range(len(references)):
                emb_pairs.append([tmp_query_list[i], references[j]])
        # compute pairwise distances for the qmax system
        if num_workers > 0:
            col_scores = Parallel(n_jobs=num_workers,
                                  backend='multiprocessing')(
                delayed(compute_distance_qmax)(
                    emb_pairs[i][0],
                    emb_pairs[i][1])
                for i in range(len(emb_pairs)))
            scores = np.array(col_scores).reshape((num_segments,
                                                   len(references)))
        else:
            for idx in range(len(emb_pairs)):
                scores[(idx // len(references)), idx % len(references)] = \
                    compute_distance_qmax(emb_pairs[idx][0],
                                          references[idx][1])

    # convert distances into similarities
    scores = -scores

    # select the most similar reference for each query segment
    sel_idxs = np.argmax(scores, axis=1)

    # parse the results
    results = []
    for i in range(scores.shape[0]):
        tmp_id = sel_idxs[i]
        query_start = i * hop_in_frames / sample_rate * hop_in_samples
        query_end = (i * hop_in_frames +
                     segment_in_frames) / sample_rate * hop_in_samples
        ref_start = 0
        ref_end = ref_sizes[tmp_id] / sample_rate * hop_in_samples
        results.append(('{}'.format(query_name),
                        '{:.3f}'.format(query_start),
                        '{:.3f}'.format(query_end),
                        '{}'.format(reference_names[tmp_id]),
                        '{:.3f}'.format(ref_start),
                        '{:.3f}'.format(ref_end),
                        '{:.3f}'.format(scores[i, tmp_id])))

    # if there are multiple queries, return the results to append
    if mul_queries:
        return results
    # if there is a single query, output the csv
    else:
        pd.DataFrame(results).to_csv(output, header=None, index=None)


def get_2dftm_query(feat, onset_env):
    """
    Compute 2DFTM embedding of a query window.

    :param feat: chroma feature of the query window
    :param onset_env: onset envelope of the query window
    :return: 2DFTM embedding of the input
    """
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env,
                                       units='time')
    hop_length = int(44100 * 4096 / 44100)
    beats = np.array(
        np.round(beats * 44100 / float(hop_length))).astype(np.int)
    feat = extract_2dftm(feat, beats)
    return feat


def get_2dftm_ref(feat, filename):
    """
    Compute 2DFTM embedding of a reference song.

    :param feat: chroma feature of the reference song
    :param filename: filename of the reference song
    :return: 2DFTM embedding of the input
    """
    onset_file = './data/ASID_onsets/{}_onset.npy'.format(filename)
    onset_env = np.load(onset_file)
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env,
                                       units='time')
    hop_length = int(args.sample_rate * 4096 / 44100)
    beats = np.array(
        np.round(beats * args.sample_rate / float(hop_length))).astype(
        np.int)
    feat = extract_2dftm(feat, beats)
    return feat


def get_re_move_emb(feat, feat_name):
    """
    Compute Re-MOVE embedding of an input. If the embedding is pre-computed,
    it simply loads the pre-computed embedding.

    :param feat: chroma feature of the input
    :param feat_name: filename of the input
    :return: Re-MOVE embedding of the input
    """
    if os.path.exists(
            os.path.join('./data/jamendo_remove/{}_remove.npy'.format(
                feat_name))):
        return np.load(
            os.path.join('./data/jamendo_remove/{}_remove.npy'.format(
                feat_name)))
    else:
        model = create_re_move_model()
        feat = np.concatenate((feat, feat[:11]))
        feat = model.predict(feat.T[np.newaxis, :, :, np.newaxis])
        return feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparing similarities among a query and references.')

    parser.add_argument('-q',
                        '--query',
                        help='path to query.')
    parser.add_argument('-r',
                        '--reference',
                        help='path to reference list.')
    parser.add_argument('-out',
                        '--output',
                        help='path to store the results.')
    parser.add_argument('-s',
                        '--system',
                        type=str,
                        default='re-move',
                        choices=['qmax', '2dftm', 're-move'],
                        help='system to use')
    parser.add_argument('-sr',
                        '--sample_rate',
                        type=int,
                        default=44100,
                        help='directory to store the embeddings.')
    parser.add_argument('--segment_size',
                        default=180,
                        type=float,
                        help='segment size (in seconds) '
                             'for the identification window.')
    parser.add_argument('--hop_size',
                        default=30,
                        type=float,
                        help='hop size (in seconds) '
                             'for the identification window.')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num_of_workers for parallel processing.')

    args = parser.parse_args()
    print('Using {} system with segment size of {}s and '
          'hop size of {}s'.format(args.system,
                                   args.segment_size,
                                   args.hop_size))

    if not os.path.exists(os.path.dirname(args.output)):
        Exception('Output directory does not exist.')

    start_time = time.monotonic()

    ref_names = []
    ref_feats = []
    ref_sizes = []

    with open(args.reference) as f:
        refs = f.read().splitlines()
    for line in refs:
        if line.endswith('.npy'):
            filename = os.path.splitext(os.path.basename(line))[0]
            if filename.endswith('_crema'):
                filename = filename.split('_crema')[0]
            feat = np.load(line)
            ref_sizes.append(feat.shape[1])
            ref_feats.append(feat)
            ref_names.append(filename)

    if args.num_workers > 0:
        if args.system == 're-move':
            proc_feats = Parallel(n_jobs=args.num_workers,
                                  backend='multiprocessing')(
                delayed(get_re_move_emb)(
                    ref_feats[i], ref_names[i]) for i in range(len(ref_feats)))
            ref_feats = proc_feats
        if args.system == '2dftm':
            proc_feats = Parallel(n_jobs=args.num_workers,
                                  backend='multiprocessing')(
                delayed(get_2dftm_ref)(
                    ref_feats[i], ref_names[i]) for i in range(len(ref_feats)))
            ref_feats = proc_feats
    else:
        if args.system == 're-move':
            proc_feats = [get_re_move_emb(ref_feats[i], ref_names[i])
                          for i in range(len(ref_feats))]
            ref_feats = proc_feats
        if args.system == '2dftm':
            proc_feats = [get_2dftm_ref(ref_feats[i], ref_names[i])
                          for i in range(len(ref_feats))]
            ref_feats = proc_feats

    print('Reference features/embeddings are loaded.')

    query_paths = []
    query_names = []
    if args.query.endswith('.npy'):
        query = np.load(args.query)
        query_name = os.path.splitext(os.path.basename(args.query))[0]
        if query_name.endswith('_crema'):
            query_name = query_name.split('_crema')[0]
        else:
            query_name = query_name
        prepare_output_single_query(query, query_name, ref_feats, ref_names,
                                    args.output, args.system, args.sample_rate,
                                    args.segment_size, args.hop_size,
                                    args.num_workers, ref_sizes, False)
    elif args.query.endswith('.lst'):
        with open(args.query) as f:
            queries = f.read().splitlines()
        for line in queries:
            query_paths.append(line)
            filename = os.path.splitext(os.path.basename(line))[0]
            if filename.endswith('_crema'):
                filename = filename.split('_crema')[0]
            query_names.append(filename)
        prepare_output_mul_queries(query_paths, query_names, ref_feats,
                                   ref_names, args.output, args.system,
                                   args.sample_rate, args.segment_size,
                                   args.hop_size, args.num_workers, ref_sizes)
    else:
        Exception('Input type for query not understood.')

    total_time = time.monotonic() - start_time

    print('All the files have been processed. '
          'The entire process took {:.0f}m{:.0f}s.'.format(total_time // 60,
                                                           total_time % 60))
