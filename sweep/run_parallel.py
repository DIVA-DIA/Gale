import itertools
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
from sigopt import Connection

from template.RunMe import RunMe

# Init SigOpt Paramters ##################################################
SIGOPT_TOKEN = "NDGGFASXLCHVRUHNYOEXFYCNSLGBFNQMACUPRHGJONZYLGBZ"  # production
# SIGOPT_TOKEN = "EWODLUKIPZFBNVPCTJBQJGVMAISNLUXGFZNISBZYCPJKPSDE"  # dev
SIGOPT_FILE = "sweep/configs/sigopt_final_config_standard.json"
SIGOPT_PROJECT = "mahpd"
SIGOPT_PARALLEL_BANDWIDTH = 1

# Init System Parameters #################################################
MAX_PARALLEL_EXPERIMENTS = 16

# GPUs_LIST = range(torch.cuda.device_count())
# GPUs_LIST = [3, 4, 5, 6, 7, 8]
GPUs_LIST = [1, 2, 3, 4, 5, 6, 7]
MAX_PROCESSES_PER_GPU = 4

# CPUs_LIST = range(mp.cpu_count())
CPUs_LIST = range(5, 64)

SERVER = 'lucy'
SERVER_PREFIX = '' if SERVER == 'dana' else '/HOME/albertim'
OUTPUT_FOLDER = ('/home/albertim' if SERVER == 'dana' else  SERVER_PREFIX) + "/output_init"

# Experiment Parameters ##################################################
EXPERIMENT_NAME_PREFIX = "spectral"
EPOCHS = 60 # For CB55 is /5
SIGOPT_RUNS = 30 # 10 * num of parameters to optimize + 10 buffer + 10 top performing
MULTI_RUN = None # Use None for disabling!
RUNS_PER_VARIANCE = 20



##########################################################################

MODELS = [
    # "LDA_Simple",
    # "InitBaseline",
    # "InitBaselineVGGLike",
    # "LDApaper",
    # "babyresnet18",
    #"raieresnet18",
    "DCT_2",
    "DCT_3",
    "RND_2",
    "FFT_2",
]

DATASETS = [
    SERVER_PREFIX + "/dataset/DIVA-HisDB/classification/CB55_149",
    SERVER_PREFIX + "/dataset/HAM10000",
    # SERVER_PREFIX + "/dataset/CIFAR10",
    # SERVER_PREFIX + "/dataset/CINIC10",
    # "/var/cache/fscache/CINIC10",
    SERVER_PREFIX + "/dataset/ColorectalHist",
    SERVER_PREFIX + "/dataset/Flowers",
    # SERVER_PREFIX + "/dataset/ImageNet",
    # SERVER_PREFIX + "/dataset/signatures/GPDS-last100/genuine",
]

# (Init function, sigopt-project-id, --extra, sigopt-file)
# RUNS = [
#     # ("random",          189837, "", None),
#     ("randisco",        191969, "--trim-lda False --retrain True ", "sweep/configs/sigopt_final_config_randisco.json"),
#     # ("pure_lda",        None, "", None),
#     # ("pure_pca",        None, "", None),
#     ("pcdisc",          191970, "--trim-lda False --retrain True ", "sweep/configs/sigopt_final_config_sbgatto.json"),
#     # ("lpca",            None, "", None),
#
#     # ("mirror_lda",      None, "", None),
#     # ("highlander_lda",  None, "", None),
#     # ("greedya",         None, "--trim-lda False --retrain True ", None),
#     # ("reverse_pca",     None, "", None),
#     # ("relda",           None, "", None),
# ]

RUNS = [
    (None, None, "", None),
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
        sigopt_parallel_bandwidth,
        sigopt_runs=None,
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
        sigopt_project : str
            SigOpt project name
        sigopt_parallel_bandwidth : int
            Number of concurrent parallel optimization running
        sigopt_runs : int or None
            Number of updates of SigOpt required

        Returns
        -------
        experiments : List(Experiment)
            List of created experiments ready to be run
        """
        experiments = []
        for dataset in datasets:
            for model in models:
                for (init, experiment_id, extra, sigopt_custom_file) in runs:
                    # Construct experiment name
                    experiment_name = ExperimentsBuilder._construct_experiment_name(
                        experiment_name_prefix=experiment_name_prefix, dataset=dataset, init=init, model=model,
                    )

                    # Create an experiment and gets its ID if necessary
                    if experiment_id is None:
                        experiment_id = RunMe().create_sigopt_experiment(
                            sigopt_token=sigopt_token,
                            sigopt_file=sigopt_file if sigopt_custom_file is None else sigopt_custom_file,
                            sigopt_project=sigopt_project,
                            sigopt_runs=sigopt_runs,
                            sigopt_parallel_bandwidth=sigopt_parallel_bandwidth,
                            experiment_name=experiment_name,
                            minimize_best_epoch=False
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
                        f"--inmem "
                        f"--patches-cap 200000 "
                        f"--validation-interval 2 "
                    )
                    if multi_run is not None:
                        additional += f"--multi-run {multi_run} "

                    # Create as many parallel one as required
                    sigopt_repeat = experiment.observation_budget - experiment.progress.observation_count
                    experiments.append([Experiment(
                        experiment_name=experiment_name,
                        model_name=model,
                        output_folder=output_folder,
                        input_folder=dataset,
                        #epochs=epochs, #int(epochs/5) if "CB55" in dataset else epochs,
                        epochs=int(epochs/5) if "CB55" in dataset else epochs,
                        init=init,
                        additional=additional
                    ) for _ in range(sigopt_parallel_bandwidth + sigopt_repeat)])
        # Flatten the list of lists of experiments s.t [a,a,a,b,b,b,c,c,c] -> [a,b,c,a,b,c,a,b,c]
        return [y for x in itertools.zip_longest(*experiments) for y in x if y is not None]

    @staticmethod
    def build_variance_combinations(
        experiment_name_prefix,
        datasets,
        models,
        runs,
        runs_per_variance,
        epochs,
        output_folder,
        sigopt_token,
    ):
        """Create a set of experiments to be run for measuring the variance of the best performance

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
        runs_per_variance : int
            Number of runs for each configuration
        epochs : int
            Max epochs for each experiment
        output_folder : str
            Path to the output folder
        sigopt_token : str
            SigOpt API token

        Returns
        -------
        experiments : List(Experiment)
            List of created experiments ready to be run
        """
        conn = Connection(client_token=sigopt_token)
        conn.set_api_url("https://api.sigopt.com")

        # Fetch all experiments
        sigopt_list = []
        for experiment in conn.experiments().fetch().iterate_pages():
            sigopt_list.append(experiment)

        experiments = []
        for dataset in datasets:
            for model in models:
                for (init, experiment_id, extra, _) in runs:
                    # Construct experiment name
                    experiment_name = ExperimentsBuilder._construct_experiment_name(
                        experiment_name_prefix=experiment_name_prefix, dataset=dataset, init=init, model=model,
                    )
                    best_parameters = ExperimentsBuilder._get_best_parameters(
                        conn, sigopt_list, [experiment_name]
                    )
                    for i in range(runs_per_variance):
                        experiments.append(Experiment(
                            experiment_name=f"multi_" + experiment_name,
                            model_name=model,
                            output_folder=output_folder,
                            input_folder=dataset,
                            epochs=epochs,
                            init=init,
                            additional=(
                                f"{extra} "
                                f"--wandb-project sigopt_{experiment_name_prefix} "
                                f"-j {ExperimentProcess.num_workers():d} "
                                f"{' '.join(['--' + k + ' ' + str(v) for k, v in best_parameters.items()])} "
                            )
                        ))
        return experiments

    @staticmethod
    def _construct_experiment_name(experiment_name_prefix, dataset, init=None, model=None):
        experiment_name = experiment_name_prefix
        if init is not None:
            experiment_name += '_' + init
        if model is not None:
            experiment_name += '_' + model
        experiment_name += '_' + Path(dataset).stem
        return experiment_name

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
        init : str = None,
        additional : str = None,
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
            f"-rc ImageClassification "
            f"--experiment-name {self.experiment_name:s} "
            f"--model {self.model_name:s} "
            f"--output-folder {self.output_folder:s} "
            f"--input-folder {self.input_folder:s} "
            f"--epochs {self.epochs:d} "
        )
        if self.init is not None:
            cmd += f"--init --init-function {self.init:s} "
        if self.additional is not None:
            cmd += self.additional
        if self.gpu_index is not None:
            cmd += f" --gpu-id {self.gpu_index:d} "
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
        max_allowed = np.min([len(GPUs_LIST) * MAX_PROCESSES_PER_GPU, MAX_PARALLEL_EXPERIMENTS])
        max_needed = len(DATASETS) * len(MODELS) * len(RUNS) * SIGOPT_PARALLEL_BANDWIDTH
        return int(np.floor(len(CPUs_LIST) / np.min([max_allowed, max_needed])))

    @staticmethod
    def list_cpus(index) -> str:
        workers = ExperimentProcess.num_workers()
        start_index = index * workers
        return ",".join([str(x) for x in CPUs_LIST[start_index : start_index + workers]])


##########################################################################
def run_experiments(gpu_indexes, processes_per_gpu, queue):
    processes = []
    i = 0
    for _ in range(processes_per_gpu):
        for gpu_index in gpu_indexes:
            cpu_list = ExperimentProcess.list_cpus(index=i)
            process = ExperimentProcess(queue=queue, gpu_index=gpu_index, cpu_list=cpu_list)
            process.start()
            processes.append(process)
            # time.sleep(15)
            i += 1
            if i == MAX_PARALLEL_EXPERIMENTS:
                break
        if i == MAX_PARALLEL_EXPERIMENTS:
            break

    for p in processes:
        p.join()

##########################################################################
if __name__ == '__main__':

    # Init queue item
    queue = Queue()

    # print("sigopt...")
    # experiments = []
    # experiments.extend(ExperimentsBuilder.build_sigopt_combinations(
    #     experiment_name_prefix=EXPERIMENT_NAME_PREFIX,
    #     datasets=DATASETS,
    #     models=MODELS,
    #     runs=RUNS,
    #     multi_run=MULTI_RUN,
    #     epochs=EPOCHS,
    #     output_folder=OUTPUT_FOLDER,
    #     sigopt_token=SIGOPT_TOKEN,
    #     sigopt_file=SIGOPT_FILE,
    #     sigopt_project=SIGOPT_PROJECT,
    #     sigopt_parallel_bandwidth=SIGOPT_PARALLEL_BANDWIDTH,
    #     sigopt_runs=SIGOPT_RUNS,
    # ))
    # [queue.put(e) for e in experiments]
    # run_experiments(GPUs_LIST, MAX_PROCESSES_PER_GPU, queue)

    print("variance...")
    experiments = []
    experiments.extend(ExperimentsBuilder.build_variance_combinations(
        experiment_name_prefix=EXPERIMENT_NAME_PREFIX,
        datasets=DATASETS,
        models=MODELS,
        runs=RUNS,
        runs_per_variance=RUNS_PER_VARIANCE,
        epochs=EPOCHS,
        output_folder=OUTPUT_FOLDER,
        sigopt_token=SIGOPT_TOKEN,
    ))
    [queue.put(e) for e in experiments]
    run_experiments(GPUs_LIST, MAX_PROCESSES_PER_GPU, queue)

    print("...finished!")
