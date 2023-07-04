import datetime
from ray import tune

MAX_PARALLEL_TRIALS = 10
NUM_TRIALS          = 1

METRIC_MODE, METRIC = "min", "weighted"
EXPERIMENT_NAME     = f'tune_{METRIC}_model_batch_lr'

JOB_TIMEOUT    = 60 * 60 * 24  # in seconds (24hr) - this is time spent in the queue + running the training job
FILE_TIMEOUT   = 60 * 5        # in seconds (5min) - this is time after the job to wait for results
FLOW_TIMEOUT   = 60 * 60 * 12  # in seconds (12hr) - this is time spent in the queue + running the flow jobs
METRIC_TIMEOUT = 60 * 60 * 12  # in seconds (12hr) - this is time spent in the queue + running the metric job

TIME_BETWEEN_QUERIES = 5  # in seconds

RAY_NUM_CPUS = 2  # must use at least two - one for manager, one for trials

BASE_DIR  = "/afs/cern.ch/user/y/ymaidann/tune"
TUNE_DIR  = BASE_DIR + "/tune"
TRIAL_DIR = TUNE_DIR + "/trials"

RESULTS_FILE = BASE_DIR + '/results/condor_tune_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'

# Specify the initial values for all hyperparameters
initial = BASE_DIR + '/options_files/initial.json'

# Specify the hyperparameters, with tuning options
config = initial
config["learning_rate"] = tune.choice([0.0001, 0.001]) # both bounds inclusive
config["dropout"] = tune.choice([0.3, 0.2, 0.1])
"""
config["num_embedding_layers"] = tune.choice([4, 6, 8])
config["num_encoder_layers"] = tune.choice([4, 8])
config["num_branch_encoder_layers"] = tune.choice([4, 8])
config["num_classification_layers"] = tune.choice([2, 4, 6, 8])
config["num_attention_heads"] = tune.choice([4, 8])
config["l2_penaly"] = [0.0005, 0.0001, 0.005, 0.001]
"""