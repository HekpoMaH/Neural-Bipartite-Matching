# Neural-Bipartite-Matching

## Dependencies

All of the experiments were run in a virtual environment running Python 3.8.1 and having all of the following packages installed (obtained by `pip freeze`):
```
absl-py==0.9.0
azure-common==1.1.24
azure-nspkg==3.0.2
azure-storage==0.36.0
beautifulsoup4==4.8.2
cachetools==4.0.0
certifi==2019.11.28
cffi==1.14.0
chardet==3.0.4
cryptography==2.8
cycler==0.10.0
decorator==4.4.1
docopt==0.6.2
dpu-utils==0.2.8
google==2.0.3
google-auth==1.11.2
google-auth-oauthlib==0.4.1
googledrivedownloader==0.4
grpcio==1.27.2
h5py==2.10.0
idna==2.8
imageio==2.6.1
isodate==0.6.0
joblib==0.14.1
kiwisolver==1.1.0
llvmlite==0.31.0
Markdown==3.2.1
matplotlib==3.1.3
more-itertools==8.2.0
networkx==2.4
numba==0.48.0
numpy==1.18.2
oauthlib==3.1.0
opt-einsum==3.1.0
overrides==2.8.0
pandas==1.0.1
Pillow==7.0.0
plyfile==0.7.1
protobuf==3.11.3
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycparser==2.19
pyparsing==2.4.6
pyro-api==0.1.1
pyro-ppl==1.2.1
python-dateutil==2.8.1
pytz==2019.3
PyWavelets==1.1.1
rdflib==4.2.2
requests==2.22.0
requests-oauthlib==1.3.0
rsa==4.0
scikit-image==0.16.2
scikit-learn==0.22.1
scipy==1.4.1
seaborn==0.10.0
sentencepiece==0.1.85
SetSimilaritySearch==0.1.7
six==1.14.0
soupsieve==1.9.5
tensorboard==2.1.0
torch==1.4.0
torch-cluster==1.5.4
torch-geometric==1.4.1
torch-scatter==2.0.3
torch-sparse==0.5.1
tqdm==4.42.1
urllib3==1.25.8
Werkzeug==1.0.0
```

## Preparation
Before running any training, prepare datasets/pre-trained models.

For the data, run `./gen_all.sh`. This generates all datasets used into
the working directory. Naming convention is as follows:
`{all_iter/bfs}_less_wired` for datasets containing training/testing data *per
each iteration of the Ford-Fulkerson algorithm* (several datapoints per graph)
for augmenting path finding and BFS. `graph_only` datasets contain only input
graphs (one datapoint per graph). Datasets generated with different edge
probability than 1/4 have their numerator and denominator appended
after dataset name. Inside each dataset, raw and processed data is provided at
different scales.

Once you want to generate the extra training data for the PNA, run
`./gen_extra.sh 8 8 bfs_less_wired 1 4`.

For obtaining the pre-trained models, run `unzip models_to_test.zip`. The
folder is organised hierarchically -- first level splits on GNN architecture
type. Second level -- whether reachability is learnt. AugmentingPath
corresponds to training
without learning reachability and AugmentingPathPlusBFS with reachability. Last
level contains serialized PyTorch model per each training epoch.

## Training

For training a model, follow the `train.py` CLI (`python train.py -h`).  Pick
which algorithms to learn (`--algorithms` is a repeatable parameter), model
name (defaults to current time, if not provided) and processor type.
(Currently, bottleneck finding and augmentation of capacities are always
learnt.) Running the script will populate models into `.serialized_models`
directory, each model in the format `test_MODELNAME_epoch_EPOCH`, where
`MODELNAME` is the model name chosen and `EPOCH` is the training epoch.

## Testing

Testing mean/last step accuracy can be achieved via the `test.py` script, which
also offers a CLI.

Testing flow accuracy however is done via the `comparator.py` script (also
having CLI). Rather than a single model, the script takes a model's folder and
calculates the flow accuracy/error for each epoch. Results are populated in the
`results` folder in the format `results_MODELNAME_TERM_SCALE` for mean absolute
flow error and `results_where_the_same_MODELNAME_EDGEPROB_TERM_SCALE` for
accuracy (checks the number of times deterministic and neural gave *the same*
flow).  `TERM`, `EDGEPROB` and `SCALE` are automatically appended, based on CLI
parameters provided and are empty strings for default values. `TERM` becomes
`BFS-based`, if use BFS for termination and `X` if use threshold of integer
value X.

Since this is a lot of options, consider a few examples. If one wants
to test a pretrained model using MPNN aggregator, which has been trained both
on augmenting path finding and reachability, neurally performs the bottleneck
and capacity augmentation steps, the command is as follows:
```
python comparator.py --algorithms AugmentingPath --algorithms BFS --model-name ADESCRIPTIVEMODELNAME --use-BFS-for-termination --use-neural-bottleneck --use-neural-augmentation --processor-type MPNN models_to_test/MPNN/AugmentingPathPlusBFS
```
and it will save flow accuracy results in
`results_where_the_same_ADESCRIPTIVEMODELNAME_BFS-based.txt`, one line per
epoch, 10 numbers (runs) per epoch.

If we now want test on a 2x scale with edge probability 1/5, run:
```
python comparator.py --algorithms AugmentingPath --algorithms BFS --model-name ADESCRIPTIVEMODELNAME --use-BFS-for-termination --use-neural-bottleneck --use-neural-augmentation --processor-type MPNN models_to_test/MPNN/AugmentingPathPlusBFS --upscale _2x --probp 1 --probq 5
```
and it will save flow accuracy results in
`results_where_the_same_ADESCRIPTIVEMODELNAME_1_5_BFS-based.txt`, one line per
epoch, 10 numbers (runs) per epoch.

Note that the `--model-name` does not need to match the one for training. It
could be anything you'd like.

If one wants to test just a single model, create an empty folder and put just
a single model inside it. The resulting output file will have only 1 line of 10
numbers, one per each run.

If you want to test PNA without the STD aggregator, modify the
`pna_aggregators` field inside `hyperparameters.py` to NOT include `std`.

## Plotting

Once results are calculated, plotting can be done by following the `plot.py`
CLI. For `MODEL_NAME` use the **exact same name** `MODEL_NAME` you used when
producing test results. A figure will be created in the `figures` folder,
following the naming conventions of the testing scripts, e.g. different scale
testing is appended automatically to the end of filename.

As plotting for different edge probability distribution was not used,
it is not yet supported.

## Demo

The Jupyter notebook `demo.ipynb` contains a standalone demo, which uses
a trained model (extract models before running the demo) to find the maximum
flow on a small example. It should serve both as an illustration of how the
network executes Ford-Fulkerson and as a MWE on how to use the code.
