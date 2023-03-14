import argparse
import os
import tqdm
import time
import copy
import sys
import pickle
import re

CODE_DIR = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(CODE_DIR)

from predict import *

def load_intent_model(device='cpu'):
        # input: np array of queries [B]
    # output: list of intents [B]taskgrasp_refine
    os.chdir('../JointBERT')
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="grasp_model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--cat', default=None)
    parser.add_argument('--idx', default=None)
    parser.add_argument('--grasp_folder ', default=None)
    parser.add_argument('--sampler', default=None)
    parser.add_argument('--grasp_space', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--max_iterations', default=None)
    parser.add_argument('--method', default=None)
    parser.add_argument('--classifier', default=None)
    parser.add_argument('--experiment_type', default=None)
    parser.add_argument('--cfg', default=None)
    pred_config = parser.parse_args()

    args = get_args(pred_config)
    model = load_model(pred_config, args, device)

    os.chdir('../graspflow')
    return model

def get_intents(args=None, query=None, model=None):
    # input: np array of queries [B]
    # output: list of intents [B]taskgrasp_refine
    os.chdir('../JointBERT')
    if query is not None:
        with open("query.txt", 'w') as f:
            for line in query:
                f.write(line + "\n")
        query = "query.txt"
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="grasp_model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--cat', default=None)
    parser.add_argument('--idx', default=None)
    parser.add_argument('--grasp_folder ', default=None)
    parser.add_argument('--sampler', default=None)
    parser.add_argument('--grasp_space', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--max_iterations', default=None)
    parser.add_argument('--method', default=None)
    parser.add_argument('--classifier', default=None)
    parser.add_argument('--experiment_type', default=None)
    pred_config = parser.parse_args(["--input_file", args.command if query is None else query])


    args = get_args(pred_config)
    device = get_device(pred_config)
    _, _, intent_preds = predict(pred_config, model)

    intent_label_lst = get_intent_labels(args)

    res = []
    for i in intent_preds:
        res.append(intent_label_lst[i])

    os.chdir('../graspflow')
    return res, intent_preds