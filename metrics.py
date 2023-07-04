import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def closest_step(lst, step):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i] - step))]

def metrics(trial_dir):
    for x in os.listdir(trial_dir):
        if x.startswith('events.out.'): # this is tensorboard logger file
            tensor_logger = x
    ea = event_accumulator.EventAccumulator(tensor_logger)
    ea.Reload()

    # Save results of epoch with minimum validation loss
    df = pd.DataFrame(ea.Scalars("validation_loss/validation_loss"))
    idxmin = df["value"].idxmin()
    metrics = {"validation_loss/validation_loss": df["value"][idxmin]}

    df = pd.DataFrame(ea.Scalars("validation_accuracy"))
    metrics["validation_accuracy"] = df["value"][idxmin]

    # The step in validation loss does not directly correpond to training loss
    # Find the closest step in value
    step = df["step"][idxmin]

    df = pd.DataFrame(ea.Scalars("loss/total_loss"))
    closest_step = closest_step(df["step"], step)
    metrics["loss/total_loss"] = df["value"][df["step"] == closest_step].values[0]

    return metrics
