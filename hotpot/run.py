import os
import json
import argparse

from hotpotqa import HotPotQATask
from models import gpt_usage
from lats import mcts_search
import logging

def run(args):
    task = HotPotQATask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0
    
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    count = 0
    task_accs = []
    info = []

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        state, value, all_nodes, reward, em = mcts_search(args, task, i)

         # log main metric
        if em is None:
            em = 0
        task_accs.append(em)
        cnt_avg = sum(task_accs) / len(task_accs)
        print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
        #all_nodes_dict = [(node.to_dict(), value) for node, value in all_nodes]
        
       
    n = args.task_end_index - args.task_start_index
    print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'llama2', 'gpt-3.5-turbo-0613'], default='gpt-3.5-turbo-16k')
    args.add_argument('--temperature', type=float, default=1.0)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords', 'hotpot'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--mcts', action='store_true')
    args.add_argument('--log', type=str)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)