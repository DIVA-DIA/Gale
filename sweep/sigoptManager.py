from sigopt import Connection


def delete_with_pattern(experiment_list, name):
    for n in experiment_list:
        if name in n.name:
            conn.experiments(n.id).delete()


def retrieve_id_by_name(experiment_list, parts):
    retrieved = []
    for n in experiment_list:
        if all(p in n.name for p in parts):
            retrieved.append(n.id)
    return retrieved


def print_with_pattern(experiment_list, parts):
    dict = {}
    for n in experiment_list:
        if all(p in n.name for p in parts):
            value = conn.experiments(n.id).best_assignments().fetch()
            if value.data:
                dict[n.name] = [value.data[0].value, n.progress.observation_count]
    for i in sorted(dict):
        print('{:75}'.format(i), "\t", dict[i])
    print("Found: " + str(len(dict)) + "\n\n")


def print_with_pattern_multimetric(experiment_list, parts):
    dict = {}
    for n in experiment_list:
        if all(p in n.name for p in parts):
            efficient_results = conn.experiments(n.id).best_assignments().fetch()
            if efficient_results.data:
                dict[n.name] = [[d.value for d in data.values] for data in efficient_results.data]
    for i in sorted(dict):
        print(f'{i:75}')
        for e in dict[i]:
            print(f"\t{e}")
    print("Found: " + str(len(dict)) + "\n\n")


def print_best_assignement_with_pattern(experiment_list, parts):
    for experiment in experiment_list:
        if all(p in experiment.name for p in parts):
            value = conn.experiments(experiment.id).best_assignments().fetch()
            if value.data:
                best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
                lr = best_assignments.data[0].assignments['lr']
                weight_decay = best_assignments.data[0].assignments['weight_decay']
                # print('{:60}'.format(experiment.name), "\t",
                #       value.data[0].value, "\t",
                #       experiment.progress.observation_count, "\t",
                #       lr, "\t",
                #       weight_decay)
                print("cd {}".format(experiment.name.split("/")[0]), "\n",
                      "cd `ls`\n",
                      "cd `ls`\n",
                      "cd {}".format(experiment.name.split("/")[-1]), "\n",
                      "cd `ls`\n",
                      "cd `ls`\n",
                      "cd `ls`\n",
                      "cd `ls`\n",
                      "cd lr={}".format(lr), "\n",
                      "cd `ls`\n",
                      "cd `ls`\n",
                      "cd `ls`\n",
                      "cp -r `ls` /local/scratch/albertim/best/{}{}".format(experiment.name.split("/")[0],
                                                                            experiment.name.split("/")[-1]), "\n",
                      "cd /local/scratch/albertim/output")


if __name__ == '__main__':
    SIGOPT_TOKEN = "NDGGFASXLCHVRUHNYOEXFYCNSLGBFNQMACUPRHGJONZYLGBZ"  # production
    # SIGOPT_TOKEN = "EWODLUKIPZFBNVPCTJBQJGVMAISNLUXGFZNISBZYCPJKPSDE"  # dev

    conn = Connection(client_token=SIGOPT_TOKEN)

    # Fetch all experiments
    experiment_list = []
    for experiment in conn.experiments().fetch().iterate_pages():
        experiment_list.append(experiment)

    # print_with_pattern(experiment_list, ["omg"])
    # print_with_pattern(experiment_list, ["CSG18", "v3"])
    # print_with_pattern(experiment_list, ["CSG863", "v3"])

    print_with_pattern_multimetric(experiment_list, ["final"])

    print("--------------------------------------------")

    # print_best_assignement_with_pattern(experiment_list, ["v2", "Deep"])

    print("Done!")
