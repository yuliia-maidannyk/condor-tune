import pickle, hashlib, json
from typing import Dict, Any
from condor_tune import TUNE_DIR, TRIAL_DIR

class Torch_CPU_Unpickler(pickle.Unpickler):
    # see https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch, io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def torch_unpickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return torch_unpickle(f).load()

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


## resulting/expected folder structure:
# condor-tune-project
# ├── results
# │   ├── condor_tune_2022-03-14_01-15-45.pkl         (pickled results DataFrame)
# │   ├── ...
# ├── tune                                            (TUNE_DIR)
# │   ├── train.sh                                    (script which launches training)
# │   ├── metric.sh                                   (script which launches metric calculation)
# │   ├── trials                                      (TRIAL_DIR)
# │   │   ├── 0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a  (hash of this trial's params)
# │   │   │   ├── flows                               (flows directory, created by training script)
# │   │   │   │   ├── results                         (post-flow values)
# │   │   │   │   │   ├── 1.json                      (flow results file)
# │   │   │   │   │   ├── 2.json                      (flow results file)
# │   │   │   │   │   ├── ...
# │   │   │   │   │   └── n.json                      (flow results file)
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