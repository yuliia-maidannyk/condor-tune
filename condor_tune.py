import ray      
import htcondor 

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

import os, pathlib, json, time, pickle
from typing import Dict, Any
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

from misc import *
from config import *
from metrics import *
import json

############################################################################################
## DIRECTORY SETUP
############################################################################################

create_base_directories     = True
move_current_trials_to_old  = True

if create_base_directories: pathlib.Path(TRIAL_DIR).mkdir(parents=True, exist_ok=True)
if move_current_trials_to_old: move_trials()

############################################################################################
## DEFINE TRIAL FUNCTION
############################################################################################

def run_trial(params: Dict[Any, Any], checkpoint_dir=None) -> None:
    """Submit and wait for a condor job to finish, then report results"""

    ############################################################################################
    ## PREPARE PARAMS JSON
    ############################################################################################

    trial_hash = dict_hash(params)
    params['hash'] = trial_hash
    THIS_TRIAL_DIR = f'{TRIAL_DIR}/{trial_hash}'

    # There has to be a better way
    # tune automatically converts ints to floats, so we enforce types as needed here
    #params['modeltype']  = int(params['modeltype'])
    #params['batch_size'] = int(params['batch_size'])
    #params['epochs']     = int(params['epochs'])
    print(params, type(params['epochs']))

    pathlib.Path(THIS_TRIAL_DIR).mkdir(parents=True, exist_ok=True)
    with open(f'{THIS_TRIAL_DIR}/params.json', 'w') as f:
        json.dump(params, f) # we will read this file in the training script

    ############################################################################################
    ## SETUP HTCONDOR SCHEDD
    ############################################################################################

    schedd = htcondor.Schedd() # we use xquery since we only need to check if the result is empty
    def is_running(job_id): return not empty_gen(schedd.xquery(f'ClusterId == {job_id}', projection=["ProcId", "JobStatus"]))
    def remove_job(job_id): schedd.act(htcondor.JobAction.Remove, f'ClusterId == {job_id}')

    ############################################################################################
    ## SUBMIT TRAINING JOB
    ############################################################################################

    train_job = htcondor.Submit(
    # Same syntax as the usual condor_submit file.
    # We use Python variables here to dynamically set command line arguments
        f"""
        universe = docker
        docker_image = yuliiamaidannyk/spanet:v12
        should_transfer_files = YES
        when_to_transfer_output = ON_EXIT
        arguments = {TUNE_DIR}/train.sh $(Cluster) $(Process)

        request_gpus = 1

        log = {THIS_TRIAL_DIR}/train.log
        output = {THIS_TRIAL_DIR}/train.out
        error = {THIS_TRIAL_DIR}/train.err

        queue
        """
    # If successful, this job will write several files, including:
    # THIS_TRIAL_DIR/training_done: prescence indicates that the training job is done
    # THIS_TRIAL_DIR/run_flow.cmd: HTCondor Submit file for flow job array
    # THIS_TRIAL_DIR/flows/: a directory with one json file for each example in the validation set
    # THIS_TRIAL_DIR/training_results.pkl: results of the training job, including the trained model
    )
    result = schedd.submit(train_job, count=1)
    job_id = result.cluster()

    ############################################################################################
    ## DETECT WHEN TRAINING JOB IS DONE
    ############################################################################################
    job_start_time = time.time()
    job_timeout    = lambda : (time.time() - job_start_time) > JOB_TIMEOUT
    while is_running(job_id) and not job_timeout():
        time.sleep(TIME_BETWEEN_QUERIES)

    if job_timeout() and is_running(job_id):
        remove_job(job_id)
        raise Exception('Training took too long to complete')

    # The training script exits on error, which exits the condor job without writing training_done.
    # If it terminates normally, it will write a file called training_done, then exit the condor job.
    # We use this to detect if training was successful.
    file_start_time = time.time()
    file_timeout    = lambda : (time.time() - file_start_time) > FILE_TIMEOUT
    while not pathlib.Path(f'{THIS_TRIAL_DIR}/training_done').is_file() and not file_timeout():
        time.sleep(TIME_BETWEEN_QUERIES)

    if file_timeout(): raise Exception("Job ended but training failed to complete")

    ############################################################################################
    ## SUBMIT FLOW JOBS
    ############################################################################################

    # Each job reads a file like THIS_TRIAL_DIR/flows/42.json and
    # writes a file like THIS_TRIAL_DIR/flows/results/42.json.

    # NOTE: we use bash, because it's easier to handle than the python interface for job arrays
    flow_output = os.popen(f'cd {THIS_TRIAL_DIR} && condor_submit run_flow.cmd').read()
    flow_jobid = flow_output.split(' ')[-1].split('\n')[0].strip('.')
    # NOTE: above line may change if condor_submit changes
    
    ############################################################################################
    ## DETECT WHEN FLOW JOBS ARE DONE
    ############################################################################################

    flow_start_time = time.time()
    flow_timeout    = lambda : (time.time() - flow_start_time) > FLOW_TIMEOUT
    while is_running(flow_jobid) and not flow_timeout():
        time.sleep(TIME_BETWEEN_QUERIES)

    if flow_timeout() and is_running(flow_jobid):
        remove_job(flow_jobid)
        raise Exception('Flows took too long to complete')

    ############################################################################################
    ## REPORT RESULTS
    ############################################################################################
    """
    for x in os.listdir(THIS_TRIAL_DIR):
        if x.startswith('events.out.'): # this is tensorboard logger file
            tensor_logger = x
    
    ea = event_accumulator.EventAccumulator(tensor_logger)
    ea.Reload()

    # Save results of epoch with minimum validation loss
    df = pd.DataFrame(ea.Scalars("validation_loss/validation_loss"))
    idx = df["value"].idxmin()
    result_dict = {"validation_loss/validation_loss": df["value"][idx]}

    df = pd.DataFrame(ea.Scalars("validation_accuracy"))
    result_dict["validation_accuracy"] = df["value"][idx]

    # The step in validation loss does not directly correpond to training loss
    # Find the closest step in value
    step = df["step"][idx]
    df = pd.DataFrame(ea.Scalars("loss/total_loss"))
    closest_step = closest_step(df["step"], step)
    result_dict["loss/total_loss"] = df["value"][df["step"] == closest_step].values[0]
    """
    result_dict = metrics(THIS_TRIAL_DIR)

    tune.report(tot_loss=result_dict['loss/total_loss'],          
                val_loss=result_dict['validation_loss/validation_loss'],  
                val_acc=result_dict['validation_accuracy'])    

############################################################################################
## BEGIN TRIALS
############################################################################################

ray.init(num_cpus=RAY_NUM_CPUS)
analysis = tune.run(run_trial, config=config, name=EXPERIMENT_NAME,
                           search_alg=HyperOptSearch(points_to_evaluate=[initial],
                               metric=METRIC, mode=METRIC_MODE),
                          num_samples=NUM_TRIALS,
                raise_on_failed_trial=False,
                  resources_per_trial={'cpu': RAY_NUM_CPUS/MAX_PARALLEL_TRIALS},)

############################################################################################
## SAVE RESULTS
############################################################################################

with open(RESULTS_FILE, 'wb') as f:
    pickle.dump(analysis.results_df, f)
