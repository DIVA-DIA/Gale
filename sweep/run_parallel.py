import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import torch
from sigopt import Connection

from template.RunMe import RunMe

# Init SigOpt Paramters ##################################################
SIGOPT_TOKEN = "NDGGFASXLCHVRUHNYOEXFYCNSLGBFNQMACUPRHGJONZYLGBZ"  # production
SIGOPT_TOKEN = "EWODLUKIPZFBNVPCTJBQJGVMAISNLUXGFZNISBZYCPJKPSDE"  # dev
SIGOPT_FILE = "sweep/configs/sigopt_final_config_standard.json"
SIGOPT_PROJECT = "mahpd"
SIGOPT_PARALLEL_BANDWIDTH = 10

# Init System Parameters #################################################
# NUM_GPUs = range(torch.cuda.device_count())
# NUM_GPUs = [3, 4, 5, 6, 7, 8]
NUM_GPUs = [1, 2, 3, 4, 5, 6, 7]
# CPU_CORES = mp.cpu_count()
CPU_CORES = 30
SERVER = 'lucy'
SERVER_PREFIX = '' if SERVER == 'dana' else '/HOME/albertim'
OUTPUT_FOLDER = ('/home/albertim' if SERVER == 'dana' else  SERVER_PREFIX) + "/output_init"

# Experiment Parameters ##################################################
EXPERIMENT_NAME_PREFIX = "retrain"
EPOCHS = 50 # For CB55 is /5
SIGOPT_RUNS = 500 # 10 * num of parameters to optimize + 10 buffer + 10 top performing
MULTI_RUN = 3
# RUNS_PER_VARIANCE = 5
PROCESSES_PER_GPU = 8


##########################################################################

MODELS = [
    # "LDA_Simple",
    # "InitBaseline",
    # "InitBaselineVGGLike",
    # "LDApaper",
    "babyresnet18",
]

DATASETS = [
    # SERVER_PREFIX + "/dataset/DIVA-HisDB/classification/CB55_23",
    # SERVER_PREFIX + "/dataset/HAM10000",
    # SERVER_PREFIX + "/dataset/CIFAR10",
    SERVER_PREFIX + "/dataset/CINIC10",
    # "/var/cache/fscache/CINIC10",
    # SERVER_PREFIX + "/dataset/ColorectalHist",
    # SERVER_PREFIX + "/dataset/Flowers",
    # SERVER_PREFIX + "/dataset/ImageNet",
    # SERVER_PREFIX + "/dataset/signatures/GPDS-last100/genuine",
]

# (Init function, sigopt-project-id, --extra, sigopt-file)
RUNS = [
    ("random",          None, "", None),
    ("randisco",        None, "",                                 "sweep/configs/sigopt_final_config_randisco"),
    ("randisco",        None, "--trim-lda False --retrain True ", "sweep/configs/sigopt_final_config_sbgatto.json"),
    # ("pure_lda",        None, "", None),
    # ("pure_pca",        None, "", None),
    # ("pcdisc",          None, "", None),
    # ("lpca",            None, "", None),

    # ("mirror_lda",      None, "", None),
    # ("highlander_lda",  None, "", None),
    # ("greedya",         None, "", None),
    # ("reverse_pca",     None, "", None),
    # ("relda",           None, "", None),
]

##########################################################################
# Creating Experiments
##########################################################################
class ExperimentsBuilder(object):

    @staticmethod
    def build_sigopt_combinations(
            experiment_name_prefix,
            datasets,
            models,
            runs,
            multi_run,
            epochs,
            output_folder,
            sigopt_token,
            sigopt_file,
            sigopt_project,
            sigopt_runs,
            sigopt_parallel_bandwidth,
    ):
        """Create a set of experiments to be run in parallel given the configurations

        Parameters
        ----------
        experiment_name_prefix : str
            String to prefix to the experiment name
        datasets : List(str)
            List of paths to datasets to be used
        models : List(str)
            List of models names to be used
        runs : List
            List of runs of type List[Tuple[str, int, str]] as:
             (--init-function, --sigopt-experiment-id, "string with extra parameter for this run only")
        multi_run : int
            Number of multi-run to perform
        epochs : int
            Max epochs for each experiment
        output_folder : str
            Path to the output folder
        sigopt_token : str
            SigOpt API token
        sigopt_file : str
            Path to a JSON file containing sig_opt variables and sig_opt bounds.
        sigopt_runs : int
            Number of updates of SigOpt required
        sigopt_project : str
            SigOpt project name
        sigopt_parallel_bandwidth : int
            Number of concurrent parallel optimization running

        Returns
        -------
        experiments : List(Experiment)
            List of created experiments ready to be run
        """
        experiments = []
        for dataset in datasets:
            for model in models:
                for (init, experiment_id, extra, sigopt_custom_file) in runs:
                    experiment_name = experiment_name_prefix + '_' + init + '_' + Path(dataset).stem

                    if extra is not "":
                        experiment_name = experiment_name_prefix + '_' + init + '_retrain_' + Path(dataset).stem

                    # Create an experiment and gets its ID if necessary
                    if experiment_id is None:
                        experiment_id = RunMe().create_sigopt_experiment(
                            sigopt_token=sigopt_token,
                            sigopt_file=sigopt_file if sigopt_custom_file is None else sigopt_custom_file,
                            sigopt_project=sigopt_project,
                            sigopt_runs=sigopt_runs,
                            sigopt_parallel_bandwidth=sigopt_parallel_bandwidth,
                            experiment_name=experiment_name,
                            minimize_best_epoch=True
                        )

                    # Delete open suggestions if any
                    conn = Connection(client_token=sigopt_token)
                    experiment = conn.experiments(experiment_id).fetch()
                    conn.experiments(experiment.id).suggestions().delete(state="open")

                    # Setup the additional parameters (not default ones)
                    additional = (
                        f"{extra} "
                        # f"--wandb-project sigopt_{experiment_name_prefix}_{Path(dataset).stem} "
                        f"--wandb-project sigopt_{experiment_name_prefix} "
                        f"--sigopt-token {sigopt_token:s} "
                        f"--sigopt-experiment-id {experiment_id} "
                        f"-j {ExperimentProcess.num_workers():d} "
                        f"--multi-run {multi_run} "
                        f"--inmem "
                        f"--patches-cap 200000 "
                        f"--validation-interval 2 "
                    )

                    # Create as many parallel one as required
                    for _ in range(sigopt_parallel_bandwidth):
                        experiments.append(Experiment(
                            experiment_name=experiment_name,
                            model_name=model,
                            output_folder=output_folder,
                            input_folder=dataset,
                            epochs=epochs, #int(epochs/5) if "CB55" in dataset else epochs,
                            init=init,
                            additional=additional
                        ))
        return experiments

    # @staticmethod
    # def build_variance_combinations():
    #     conn = Connection(client_token=SIGOPT_TOKEN)
    #     conn.set_api_url("https://api.sigopt.com")
    #
    #     # Fetch all experiments
    #     sigopt_list = []
    #     for experiment in conn.experiments().fetch().iterate_pages():
    #         sigopt_list.append(experiment)
    #
    #     experiments = []
    #     for dataset in DATASETS:
    #         for model in MODELS:
    #             for (init, experiment_id, extra) in RUNS:
    #                 experiment_name = EXPERIMENT_NAME_PREFIX + '_' + init + '_' + Path(dataset).stem
    #                 best_parameters = ExperimentsBuilder._get_best_parameters(
    #                     conn, sigopt_list, [experiment_name]
    #                 )
    #                 for i in range(RUNS_PER_VARIANCE):
    #                     experiments.append(Experiment(
    #                         experiment_name=f"multi_" + experiment_name,
    #                         model_name=model,
    #                         output_folder=OUTPUT_FOLDER,
    #                         input_folder=dataset,
    #                         epochs=EPOCHS,
    #                         init=init,
    #                         additional=(
    #                             f"{extra} "
    #                             f"--wandb-project sigopt_{EXPERIMENT_NAME_PREFIX}_{Path(dataset).stem} "
    #                             f"-j {ExperimentsBuilder.num_workers():d} "
    #                             f"{' '.join(['--' + k + ' ' + str(v) for k, v in best_parameters.items()])} "
    #                         )
    #                     ))
    #     return experiments

    @staticmethod
    def _retrieve_id_by_name(sigopt_list, parts):
        retrieved = []
        for experiment in sigopt_list:
            if all(p in experiment.name for p in parts):
                retrieved.append(experiment.id)
        return retrieved

    @staticmethod
    def _get_best_parameters(conn, sigopt_list, parts):
        experiment_id = ExperimentsBuilder._retrieve_id_by_name(sigopt_list, parts)
        if not experiment_id:
            print("Experiments not found")
            sys.exit(-1)
        # Select the experiments with the highest score
        scores = [conn.experiments(ID).best_assignments().fetch().data[0].value for ID in experiment_id]
        experiment_id = experiment_id[scores.index(max(scores))]
        # Return the assignments
        return conn.experiments(experiment_id).best_assignments().fetch().data[0].assignments


##########################################################################
# Experiments Object
##########################################################################
class Experiment(object):
    def __init__(
            self,
            experiment_name : str,
            model_name : str,
            output_folder : str,
            input_folder : str,
            epochs : int ,
            init : str,
            additional : str,
            gpu_index : int = None,
            cpu_list : str = None,
    ):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.output_folder = output_folder
        self.input_folder = input_folder
        self.epochs = epochs
        self.init = init
        self.additional = additional
        self.gpu_index = gpu_index
        self.cpu_list = cpu_list

    def get_cmd(self):
        cmd = ""
        if self.cpu_list is not None:
            cmd += f"taskset --cpu-list {self.cpu_list:s} "
        cmd += (
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


##########################################################################
# Running Experiments
##########################################################################
class ExperimentProcess(Process):
    def __init__(self, queue, gpu_index, cpu_list):
        super().__init__()
        self.queue = queue
        self.gpu_index = gpu_index
        self.cpu_list = cpu_list

    def run(self):
        while not self.queue.empty():
            experiment = self.queue.get()
            experiment.gpu_index = self.gpu_index
            experiment.cpu_list = self.cpu_list
            os.system(experiment.get_cmd())

    @staticmethod
    def num_workers() -> int:
        # 'max_' here is meant as "running at the same time"
        max_allowed = len(NUM_GPUs) * PROCESSES_PER_GPU
        max_needed = len(DATASETS) * len(MODELS) * len(RUNS) * SIGOPT_PARALLEL_BANDWIDTH
        return int(np.floor(CPU_CORES / np.min([max_allowed, max_needed])))

    @staticmethod
    def list_cpus(index) -> str:
        workers = ExperimentProcess.num_workers()
        start_index = 30 + index * workers
        # if start_index + workers > CPU_CORES:
        #     raise EnvironmentError(
        #         f"Attempt to allocate more cores ({start_index + workers}) than available ({CPU_CORES})."
        #     )
        return ",".join([str(x) for x in list(range(start_index, start_index + workers))])


##########################################################################
def run_experiments(gpu_indexes, processes_per_gpu, queue):
    processes = []
    max_processes = queue.qsize()
    i = 0
    for _ in range(processes_per_gpu):
        for gpu_index in gpu_indexes:
            cpu_list = ExperimentProcess.list_cpus(index=i)
            process = ExperimentProcess(queue=queue, gpu_index=gpu_index, cpu_list=cpu_list)
            process.start()
            processes.append(process)
            time.sleep(60)
            i += 1
            if i == max_processes:
                # This happens if queue.qsize() < #num process that can be allocated in total
                break
        if i == max_processes:
            break

    for p in processes:
        p.join()

##########################################################################
if __name__ == '__main__':

    # Init queue item
    queue = Queue()

    print("sigopt...")
    experiments = []
    experiments.extend(ExperimentsBuilder.build_sigopt_combinations(
        experiment_name_prefix=EXPERIMENT_NAME_PREFIX,
        datasets=DATASETS,
        models=MODELS,
        runs=RUNS,
        multi_run=MULTI_RUN,
        epochs=EPOCHS,
        output_folder=OUTPUT_FOLDER,
        sigopt_token=SIGOPT_TOKEN,
        sigopt_file=SIGOPT_FILE,
        sigopt_project=SIGOPT_PROJECT,
        sigopt_runs=SIGOPT_RUNS,
        sigopt_parallel_bandwidth=SIGOPT_PARALLEL_BANDWIDTH,
    ))
    [queue.put(e) for e in experiments]
    run_experiments(NUM_GPUs, PROCESSES_PER_GPU, queue)

    # print("variance...")
    # experiments = []
    # experiments.extend(ExperimentsBuilder.build_variance_combinations())
    # [queue.put(e) for e in experiments]
    # run_experiments(NUM_GPUs, PROCESSES_PER_GPU, queue)

    print("...finished!")
