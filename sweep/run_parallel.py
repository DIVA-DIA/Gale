import multiprocessing as mp
import os
import sys
from multiprocessing import Process, Queue
import numpy as np
import torch
from sigopt import Connection
from pathlib import Path
# Init SigOpt Paramters ##################################################
# SIGOPT_TOKEN = "NDGGFASXLCHVRUHNYOEXFYCNSLGBFNQMACUPRHGJONZYLGBZ"  # production
SIGOPT_TOKEN = "EWODLUKIPZFBNVPCTJBQJGVMAISNLUXGFZNISBZYCPJKPSDE"  # dev
SIGOPT_FILE = "sweep/sigopt_sweep_config.json"
SIGOPT_PROJECT = "init"

# Init System Parameters #################################################
NUM_GPUs = range(torch.cuda.device_count())
#NUM_GPUs = [3, 4, 5, 6, 7]
CPU_CORES = mp.cpu_count()

# Experiment Parameters ##################################################
EXPERIMENT_NAME_PREFIX = "init_v1"
OUTPUT_FOLDER = "/HOME/albertim/output_init"
EPOCHS = 20 # For CB55 is /10
RUNS_PER_INSTANCE = 3 # 150 # 10+ * num of parameters to optimize usually
RUNS_PER_VARIANCE = 2
PROCESSES_PER_GPU = 1

##########################################################################

MODELS = [
    "LDA_Simple",
]

DATASETS = [
    "/HOME/albertim/dataset/DIVA-HisDB/CB55",
    # "/HOME/albertim/dataset/HAM10000",
    # "/HOME/albertim/dataset/CIFAR10",
    # "/HOME/albertim/dataset/ColorectalHist",
    # "/HOME/albertim/dataset/Flowers",
    # "/HOME/albertim/dataset/ImageNet",
    # "/HOME/albertim/dataset/signatures/GPDS-last100/genuine",
]

INIT = [
    "random",
    # "pure_lda",
    #"mirror_lda",
    #"highlander_lda",
    #"pure_pca",
    #"lpca",
    #"reverse_pca",
    #"relda",
]

##########################################################################
# Creating Experiments
##########################################################################
class Experiment(object):
    def __init__(
        self,
        experiment_name_prefix : str,
        model_name : str,
        output_folder : str,
        input_folder : str,
        epochs : int ,
        init : str,
        additional : str,
        gpu_index : int = None,
    ):
        self.experiment_name = experiment_name_prefix + '_' + init + '_' + Path(input_folder).stem
        self.model_name = model_name
        self.output_folder = output_folder
        self.input_folder = input_folder
        self.epochs = int(epochs/10) if "CB55" in input_folder else epochs
        self.init = init
        self.additional = additional
        self.gpu_index = gpu_index

    def get_cmd(self):
        cmd = (
            f"python template/RunMe.py --ignoregit --disable-dataset-integrity "
            f"-rc ImageClassification --init --nesterov "            
            f"--experiment-name {self.experiment_name:s} "
            f"--model {self.model_name:s} "
            f"--output-folder {self.output_folder:s} "
            f"--input-folder {self.input_folder:s} "
            f"--epochs {self.epochs:d} "
            f"--init-function {self.init:s} "
        )
        if self.gpu_index is not None:
            cmd += f" --gpu-id {self.gpu_index:d} "
        if self.additional:
            cmd += self.additional
        return cmd

    def __repr__(self):
        return self.get_cmd()


class ExperimentsBuilder(object):

    @staticmethod
    def build_sigopt_combinations():
        experiments = []
        for dataset in DATASETS:
            for model in MODELS:
                for init in INIT:
                    experiments.append(Experiment(
                        experiment_name_prefix=EXPERIMENT_NAME_PREFIX,
                        model_name=model,
                        output_folder=OUTPUT_FOLDER,
                        input_folder=dataset,
                        epochs=EPOCHS,
                        init=init,
                        additional = (
                            f"--wandb-project sigopt_{Path(dataset).stem}_{init} "
                            f"-j {ExperimentsBuilder.num_workers():d} "
                            f"--sig-opt-token {SIGOPT_TOKEN:s} "
                            f"--sig-opt-runs {str(RUNS_PER_INSTANCE):s} "
                            f"--sig-opt-project {SIGOPT_PROJECT:s} "
                            f"--sig-opt {SIGOPT_FILE} "
                        )
                    ))
        return experiments

    @staticmethod
    def build_variance_combinations():
        conn = Connection(client_token=SIGOPT_TOKEN)
        conn.set_api_url("https://api.sigopt.com")

        # Fetch all experiments
        sigopt_list = []
        for experiment in conn.experiments().fetch().iterate_pages():
            sigopt_list.append(experiment)

        experiments = []
        for dataset in DATASETS:
            for model in MODELS:
                for init in INIT:
                    best_parameters = ExperimentsBuilder._get_best_parameters(
                        conn, sigopt_list, [model, dataset, init]
                    )
                    experiments.append(Experiment(
                        experiment_name_prefix=f"multi_" + EXPERIMENT_NAME_PREFIX,
                        model_name=model,
                        output_folder=OUTPUT_FOLDER,
                        input_folder=dataset,
                        epochs=EPOCHS,
                        init=init,
                        additional=(
                            f"--wandb-project sigopt_variance_{Path(dataset).stem} "
                            f"-j {ExperimentsBuilder.num_workers():d} "
                            f"--multi-run {RUNS_PER_VARIANCE:d} "
                            # TODO add all param
                            f"--lr {best_parameters['lr']:f} "
                            f"--weight-decay {best_parameters['weight_decay']:f} "
                        )
                    ))
        return experiments

    @staticmethod
    def num_workers() -> int:
        # 'max_' here is meant as "running at the same time"
        max_allowed = len(NUM_GPUs) * PROCESSES_PER_GPU
        max_needed = len(DATASETS) * len(MODELS) * len(INIT)
        return int(np.floor(CPU_CORES / np.min([max_allowed, max_needed])))

    @staticmethod
    def _retrieve_id_by_name(sigopt_list, parts):
        retrieved = []
        for experiment in sigopt_list:
            if all(p in experiment.name for p in parts):
                retrieved.append(experiment.id)
        return retrieved

    @staticmethod
    def _get_best_parameters(conn, sigopt_list, parts):
        EXPERIMENT_ID = ExperimentsBuilder._retrieve_id_by_name(sigopt_list, parts)
        if not EXPERIMENT_ID:
            print("Experiments not found")
            sys.exit(-1)
        # Select the experiments with the highest score
        scores = [conn.experiments(ID).best_assignments().fetch().data[0].value for ID in EXPERIMENT_ID]
        EXPERIMENT_ID = EXPERIMENT_ID[scores.index(max(scores))]
        # Return the assignments
        return conn.experiments(EXPERIMENT_ID).best_assignments().fetch().data[0].assignments

##########################################################################
# Running Experiments
##########################################################################
class ExperimentProcess(Process):
    def __init__(self, queue, gpu_idx):
        super().__init__()
        self.gpu_index = gpu_idx
        self.queue = queue

    def run(self):
        while not self.queue.empty():
            experiment = self.queue.get()
            experiment.gpu_index = self.gpu_index
            os.system(experiment.get_cmd())


def run_experiments(number_gpus, processes_per_gpu, queue):
    processes = []
    for _ in range(processes_per_gpu):
        for gpu in number_gpus:
            process = ExperimentProcess(queue=queue, gpu_idx=gpu)
            process.start()
            processes.append(process)
    for p in processes:
        p.join()


if __name__ == '__main__':

    # Init queue item
    queue = Queue()

    print("sigopt...")
    experiments = []
    experiments.extend(ExperimentsBuilder.build_sigopt_combinations())
    # experiments.extend(ExperimentsBuilder.build_sigopt_combinations(
    #     ["RNDBidir"],["/local/scratch/Flowers"], EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    # ))
    [queue.put(e) for e in experiments]
    run_experiments(NUM_GPUs, PROCESSES_PER_GPU, queue)

    print("variance...")
    experiments = []
    experiments.extend(ExperimentsBuilder.build_variance_combinations())
    # experiments.extend(ExperimentsBuilder.build_variance_combinations(
    #     ["FFTBidir"],["/local/scratch/HAM10000"], EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    # ))
    [queue.put(e) for e in experiments]
    run_experiments(NUM_GPUs, PROCESSES_PER_GPU, queue)

    print("...finished!")
