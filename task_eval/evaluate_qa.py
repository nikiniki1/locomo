import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm
import argparse
from global_methods import set_openai_key, set_anthropic_key
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.gpt_utils import get_gpt_answers
from task_eval.hf_llm_utils import init_hf_model, get_hf_answers

import numpy as np

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    if 'gpt' in args.model:
        set_openai_key()
    elif 'claude' in args.model:
        set_anthropic_key()
    elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
        hf_pipeline, hf_model_name = init_hf_model(args)
    else:
        raise NotImplementedError

    samples = json.load(open(args.data_file))
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)

    if os.path.exists(args.out_file) and not args.overwrite:
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    done_ids = set(out_samples.keys())

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_out_file = args.out_file + ".tmp"

    for idx, data in enumerate(samples):
        sample_id = data['sample_id']

        if sample_id in done_ids and not args.overwrite:
            continue

        print(f'Current sample: {idx} (sample_id={sample_id})')

        try:
            out_data = {'sample_id': sample_id}
            if sample_id in out_samples:
                out_data['qa'] = out_samples[sample_id]['qa'].copy()
            else:
                out_data['qa'] = data['qa'].copy()

            if 'gpt' in args.model:
                answers = get_gpt_answers(data, out_data, prediction_key, args)
            elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
                answers = get_hf_answers(data, out_data, args, hf_pipeline, hf_model_name)
            else:
                raise NotImplementedError

            exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)

            for j in range(len(answers['qa'])):
                answers['qa'][j][model_key + '_f1'] = round(exact_matches[j], 3)
                if args.use_rag and len(recall) > 0:
                    answers['qa'][j][model_key + '_recall'] = round(recall[j], 3)

            out_samples[sample_id] = answers
            done_ids.add(sample_id)

            with open(tmp_out_file, 'w') as f:
                json.dump(list(out_samples.values()), f, indent=2)
            os.replace(tmp_out_file, args.out_file)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error on sample_id={sample_id}: {repr(e)}")
            continue

    analyze_aggr_acc(
        args.data_file,
        args.out_file,
        args.out_file.replace('.json', '_stats.json'),
        model_key,
        model_key + '_f1',
        rag=args.use_rag
    )

main()