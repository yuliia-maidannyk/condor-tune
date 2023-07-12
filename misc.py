import pickle, hashlib, json
from typing import Dict, Any
from config import TUNE_DIR, TRIAL_DIR

class Torch_CPU_Unpickler(pickle.Unpickler):
    # see https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def torch_unpickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return Torch_CPU_Unpickler(f).load()

def empty_gen(generator):
    o = object() # see https://stackoverflow.com/a/21525143/13073379
    return next(generator, o) is o

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    # We need to sort arguments so {'a': 1, 'b': 2} has
    # the same hash as {'b': 2, 'a': 1}
    dhash = hashlib.md5()
    dhash.update(json.dumps(dictionary, sort_keys=True).encode())
    return dhash.hexdigest()

def move_trials():
    import shutil, os
    # move all contents of tune/trials/* to tune/old
    for dir in os.listdir(TRIAL_DIR):
        # NOTE: if dir already exists in old, we remove it and overwrite
        if os.path.exists(f"{TUNE_DIR}/old/{dir}"):
            shutil.rmtree(f"{TUNE_DIR}/old/{dir}")
        shutil.move(f"{TRIAL_DIR}/{dir}", f"{TUNE_DIR}/old/{dir}")

def convert_to_int(params):
    params["hidden_dim"] = int(params["hidden_dim"])
    params["transformer_dim"] = int(params["transformer_dim"])
    params["initial_embedding_dim"] = int(params["initial_embedding_dim"])
    params["position_embedding_dim"] = int(params["position_embedding_dim"])
    params["num_embedding_layers"] = int(params["num_embedding_layers"])
    params["num_encoder_layers"] = int(params["num_encoder_layers"])
    params["num_branch_embedding_layers"] = int(params["num_branch_embedding_layers"])
    params["num_branch_encoder_layers"] = int(params["num_branch_encoder_layers"])
    params["num_jet_embedding_layers"] = int(params["num_jet_embedding_layers"])
    params["num_jet_encoder_layers"] = int(params["num_jet_encoder_layers"])
    params["num_detector_layers"] = int(params["num_detector_layers"])
    params["num_regression_layers"] = int(params["num_regression_layers"])
    params["num_classification_layers"] = int(params["num_classification_layers"])
    params["split_symmetric_attention"] = int(params["split_symmetric_attention"])
    params["num_attention_heads"] = int(params["num_attention_heads"])
    params["skip_connections"] = int(params["skip_connections"])
    params["initial_embedding_skip_connections"] = int(params["initial_embedding_skip_connections"])
    params["linear_prelu_activation"] = int(params["linear_prelu_activation"])
    params["normalize_features"] = int(params["normalize_features"])
    params["limit_to_num_jets"] = int(params["limit_to_num_jets"])
    params["balance_particles"] = int(params["balance_particles"])
    params["balance_jets"] = int(params["balance_jets"])
    params["balance_classifications"] = int(params["balance_classifications"])
    params["partial_events"] = int(params["partial_events"])
    params["dataset_randomization"] = int(params["dataset_randomization"])
    params["batch_size"] = int(params["batch_size"])
    params["num_dataloader_workers"] = int(params["num_dataloader_workers"])
    params["mask_sequence_vectors"] = int(params["mask_sequence_vectors"])
    params["learning_rate_cycles"] = int(params["learning_rate_cycles"])
    params["balance_losses"] = int(params["balance_losses"])
    params["epochs"] = int(params["epochs"])
    params["num_gpu"] = int(params["num_gpu"])
    params["verbose_output"] = int(params["verbose_output"])
    return params
           
## resulting/expected folder structure:
# condor-tune-project
# ├── results                                         (directory for tuning results, created by user)
# │   ├── condor_tune_2022-03-14_01-15-45.pkl         (pickled results DataFrame, created by condor_tune.py)
# │   ├── ...
# ├── tune                                            (TUNE_DIR, created by condor_tune.py)
# │   ├── train.sh                                    (script which launches training)
# │   ├── metric.sh                                   (script which launches metric calculation)
# │   ├── trials                                      (TRIAL_DIR, created by condor_tune.py)
# │   │   ├── 0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a  (hash of this trial's params, created by condor_tune.py)
# │   │   │   ├── flows                               (flows directory, created by training script)
# │   │   │   │   ├── results                         (post-flow values, created by flow script)
# │   │   │   │   │   ├── 1.json                      (flow results file, created by flow script)
# │   │   │   │   │   ├── 2.json                      (flow results file, created by flow script)
# │   │   │   │   │   ├── ...
# │   │   │   │   │   └── n.json                      (flow results file, created by flow script)
# │   │   │   │   ├── 1.json                          (flow input file, created by training script)
# │   │   │   │   ├── 2.json                          (flow input file, created by training script)
# │   │   │   │   ├── ...
# │   │   │   │   └── n.json                          (flow input file, created by training script)
# │   │   │   ├── params.json                         (params for this run, chosen by HyperOpt)
# │   │   │   ├── train.log                           (training log file from HTCondor)
# │   │   │   ├── train.out                           (train stdout file from HTCondor)
# │   │   │   ├── train.err                           (train stderr file from HTCondor)
# │   │   │   ├── metric.log                          (metric log file from HTCondor)
# │   │   │   ├── metric.out                          (metric stdout file from HTCondor)
# │   │   │   ├── metric.err                          (metric stderr file from HTCondor)
# │   │   │   ├── training_done                       (flag indicating training finished successfully)
# │   │   │   ├── run_flow.cmd                        (command used to run flow job array, created by training script)
# │   │   │   ├── training_results.pkl                (pickled training results, including model weights and metrics)
# │   │   │   └── results.pkl                         (pickled final results, created by metric script)
# │   │   ├── ...
# │   ├── old
# │   │   ├── 1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b  (old trial hash)
# │   │   │   └── ... (same structure as above)
# │   │   └── ...