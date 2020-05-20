"""
Usage:
    comparator.py [options] [--algorithms=ALGO]... MODEL_DIR

Options:
    -h --help                        Show this screen.
    --algorithms ALGO                Which algorithms to add. Any of {AugmentingPath, BFS}
    --upscale UP                     Test on larger data. Remember to add underscore (e.g. _2x) [default: ]
    --model-name NAME                Specific name of model. Used for filename to save results. [default: ]
    --use-BFS-for-termination        Use BFS for deciding if more augmenting paths exist. Remember to load BFS algorithm [default: False]
    --use-neural-bottleneck          Use the network to extract the bottleneck [default: False]
    --processor-type PROC            Type of processor. One of {MPNN, PNA, GAT}. [default: MPNN]
    --use-neural-augmentation        Use the network to provide the new forward capacities after augmenting the flow. (Backward capacity is total edge capacity minus forward) [default: False]
    --probp P                        Probability P (P/Q) wired factor [default: 1]
    --probq Q                        Probability Q (P/Q) wired factor [default: 4]
    --recache                        Refresh the cache of real maximum flows [default: False]
"""
import os
import re
import torch
from docopt import docopt

import models
import utils
from hyperparameters import get_hyperparameters
from half_deterministic import run
from flow_datasets import SingleIterationDataset

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def get_max_flow(f, command):
    command = command.format(f)
    os.system(command)
    maxflow = 0
    with open("o.txt") as mf:
        maxflow = float(mf.readline())
    return maxflow

def get_algorithms_suffix(algorithms):
    suffix = ""
    for algorithm in algorithms:
        suffix += "--algorithms " + algorithm + " "
    return suffix

def write_real_maxflow_to_cache(maxflow, upscale, probp, probq):
    probpath = '' if probp == 1 and probq == 4 else '_'+str(probp)+'_'+str(probq)
    with open("./results/real_maxflow"+probpath+upscale+".txt", "w") as f:
        f.write(" ".join(str(mf) for mf in maxflow)+"\n")

def read_real_maxflow_from_cache(upscale, recache, probp, probq):
    probpath = '' if probp == 1 and probq == 4 else '_'+str(probp)+'_'+str(probq)
    try:
        if recache:
            raise FileNotFoundError

        with open("./results/real_maxflow"+probpath+upscale+".txt", "r") as f:
            return [float(x) for x in f.readline().split()]
    except FileNotFoundError:
        maxflow_real = []
        command_real = "python deterministic.py < all_iter_less_wired" + probpath + "/raw/test" + upscale + "/{} --eval_processed_input > o.txt"
        for f in sorted(os.listdir("./all_iter_less_wired" + probpath + "/raw/test" + upscale + ""), key=alphanum_key):
            mfr = get_max_flow(f, command_real)
            maxflow_real.append(mfr)
        write_real_maxflow_to_cache(maxflow_real, upscale, probp, probq)
        return maxflow_real



if __name__ == "__main__":
    args = docopt(__doc__)
    if "--model-name" in args:
        args["--model-name"] = "_" + args["--model-name"]
    upscale = args["--upscale"]
    use_bfs = args["--use-BFS-for-termination"]
    args["--use-ints"] = True # Always uses integers
    args["--probp"] = int(args["--probp"])
    args["--probq"] = int(args["--probq"])
    algorithms_suffix = get_algorithms_suffix(args["--algorithms"])

    maxflow_real = read_real_maxflow_from_cache(upscale, args["--recache"], args["--probp"], args["--probq"])
    print(maxflow_real)

    thresholds = ["BFS-based"] if use_bfs else [1, 3, 5]
    for threshold in thresholds:
        with open("./results/results_where_the_same"+args["--model-name"]+upscale+"_{}.txt".format(threshold), "w") as f:
            pass
        with open("./results/results"+args["--model-name"]+upscale+"_{}.txt".format(threshold), "w") as f:
            pass

    probpath = '' if args["--probp"] == 1 and args["--probq"] == 4 else '_'+str(args["--probp"])+'_'+str(args["--probq"])
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    processor = models.AlgorithmProcessor(DIM_LATENT, SingleIterationDataset, args["--processor-type"]).to(DEVICE)
    processor.eval()
    utils.load_algorithms(args["--algorithms"], processor, args["--use-ints"])
    print(processor)

    for threshold in thresholds:
        for model in list(sorted(os.listdir("./"+args["MODEL_DIR"]), key=alphanum_key)):
            print(threshold, model)
            processor.load_state_dict(torch.load("./"+args["MODEL_DIR"]+"/"+model))
            processor.eval()
            to_write = []
            to_write_abs = []
            with torch.no_grad():
                result_maxflows = run(args, threshold, processor, int(args["--probp"]), int(args["--probq"]), savefile=False)
            for maxflow_neural in result_maxflows:
                abs_difference = [0 for _ in range(len(maxflow_neural))]
                accuracy_values = [0 for _ in range(len(maxflow_neural))]
                for i, _ in enumerate(maxflow_neural):
                    abs_difference[i] = int(abs(maxflow_real[i] - maxflow_neural[i]))
                    accuracy_values[i] = int(maxflow_real[i] == maxflow_neural[i])
                    assert maxflow_real[i] >= maxflow_neural[i], (i, maxflow_real, maxflow_neural)
                to_write.append(str(sum(accuracy_values)/len(accuracy_values)))
                to_write_abs.append(str(sum(abs_difference)/len(abs_difference)))

            with open("./results/results"+args["--model-name"]+probpath+upscale+"_{}.txt".format(threshold), "a+") as f:
                print(*to_write_abs, sep=' ', end='\n', file=f)
            with open("./results/results_where_the_same"+args["--model-name"]+probpath+upscale+"_{}.txt".format(threshold), "a+") as f:
                print(*to_write, sep=' ', end='\n', file=f)
