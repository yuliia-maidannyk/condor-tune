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
# initial = BASE_DIR + '/options_files/initial.json'
initial = {
    "hidden_dim": 128,
    "transformer_dim": 128,
    "transformer_dim_scale": 2.0,
    "initial_embedding_dim": 16,
    "position_embedding_dim": 32,
    "num_embedding_layers": 8,
    "num_encoder_layers": 6,
    "num_branch_embedding_layers": 5,
    "num_branch_encoder_layers": 5,
    "num_jet_embedding_layers": 0,
    "num_jet_encoder_layers": 1,
    "num_detector_layers": 1,
    "num_regression_layers": 1,
    "num_classification_layers": 4,
    "split_symmetric_attention": 1,
    "num_attention_heads": 4,
    "transformer_activation": "gelu",
    "skip_connections": 1,
    "initial_embedding_skip_connections": 1,
    "linear_block_type": "GRU",
    "transformer_type": "Gated",
    "linear_activation": "gelu",
    "normalization": "LayerNorm",
    "masking": "Multiplicative",
    "linear_prelu_activation": 1,
    "event_info_file": "/afs/cern.ch/user/y/ymaidann/data/tth.yaml",
    "training_file": "/afs/cern.ch/user/y/ymaidann/data/tth.h5",
    "validation_file": "",
    "testing_file": "",
    "normalize_features": 1,
    "limit_to_num_jets": 0,
    "balance_particles": 1,
    "balance_jets": 0,
    "balance_classifications": 0,
    "partial_events": 1,
    "dataset_limit": 1.0,
    "dataset_randomization": 0,
    "train_validation_split": 0.95,
    "batch_size": 1024,
    "num_dataloader_workers": 4,
    "mask_sequence_vectors": 1,
    "combine_pair_loss": "min",
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "focal_gamma": 1.0,
    "combinatorial_scale": 0.0,
    "learning_rate_warmup_epochs": 1.0,
    "learning_rate_cycles": 1,
    "assignment_loss_scale": 1.0,
    "detection_loss_scale": 0.0,
    "kl_loss_scale": 1.0,
    "regression_loss_scale": 0.0,
    "classification_loss_scale": 0.0,
    "balance_losses": 1,
    "l2_penalty": 0.0002,
    "gradient_clip": 0.0,
    "dropout": 0.1,
    "epochs": 30,
    "num_gpu": 1,
    "verbose_output": 0,
    "usable_gpus": "",
    "trial_time": "",
    "trial_output_dir": "./test_output"
}

# Specify the hyperparameters, with tuning options
config = {
    "hidden_dim": 128,
    "transformer_dim": 128,
    "transformer_dim_scale": 2.0,
    "initial_embedding_dim": 16,
    "position_embedding_dim": 32,
    "num_embedding_layers": tune.quniform(1,8,1),
    "num_encoder_layers": tune.quniform(1,8,1),
    "num_branch_embedding_layers": 5,
    "num_branch_encoder_layers": tune.quniform(1,6,1),
    "num_jet_embedding_layers": 0,
    "num_jet_encoder_layers": 1,
    "num_detector_layers": 1,
    "num_regression_layers": 1,
    "num_classification_layers": tune.quniform(1,8,1),
    "split_symmetric_attention": 1,
    "num_attention_heads": tune.choice([2, 4, 8]),
    "transformer_activation": "gelu",
    "skip_connections": 1,
    "initial_embedding_skip_connections": 1,
    "linear_block_type": "GRU",
    "transformer_type": "Gated",
    "linear_activation": "gelu",
    "normalization": "LayerNorm",
    "masking": "Multiplicative",
    "linear_prelu_activation": 1,
    "event_info_file": "/afs/cern.ch/user/y/ymaidann/data/tth.yaml",
    "training_file": "/afs/cern.ch/user/y/ymaidann/data/tth.h5",
    "validation_file": "",
    "testing_file": "",
    "normalize_features": 1,
    "limit_to_num_jets": 0,
    "balance_particles": 1,
    "balance_jets": 0,
    "balance_classifications": 0,
    "partial_events": 1,
    "dataset_limit": 1.0,
    "dataset_randomization": 0,
    "train_validation_split": 0.95,
    "batch_size": 1024,
    "num_dataloader_workers": 4,
    "mask_sequence_vectors": 1,
    "combine_pair_loss": "min",
    "optimizer": "AdamW",
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "focal_gamma": 1.0,
    "combinatorial_scale": 0.0,
    "learning_rate_warmup_epochs": 1.0,
    "learning_rate_cycles": 1,
    "assignment_loss_scale": 1.0,
    "detection_loss_scale": 0.0,
    "kl_loss_scale": 1.0,
    "regression_loss_scale": 0.0,
    "classification_loss_scale": 0.0,
    "balance_losses": 1,
    "l2_penalty": tune.loguniform(1e-4, 1e-2),
    "gradient_clip": 0.0,
    "dropout": tune.uniform(0.01, 0.5),
    "epochs": 30,
    "num_gpu": 1,
    "verbose_output": 0,
    "usable_gpus": "",
    "trial_time": "",
    "trial_output_dir": "./test_output"
}
