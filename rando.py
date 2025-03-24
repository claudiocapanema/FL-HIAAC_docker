import numpy as np

def local_concept_drift_config(n_clients, n_rounds, me, concept_drift_round, seed):
    np.random.seed(seed)
    concept_drift_rounds = np.random.choice(n_rounds, n_clients, replace=False)
    round_number = []
    cid_number = []
    partition_number = []
    me_number = []

    for cid in range(1, n_clients + 1):

        for round_ in range(1, n_rounds + 1):

            cid_number.append(cid)
            round_number.append(round_)

            if round_ > concept_drift_round:
                partition = cid + seed

            else:
                partition = cid

            partition_number.append(partition)



    return concept_drift_rounds

def global_concept_dirft_config(ME, n_rounds, alphas, experiment_id, seed=0):
    # np.random.seed(seed)
    # fraction_min_round = 0.3
    # min_round = int(fraction_min_round * n_rounds)
    #
    # new_alphas = np.random.choice(alphas, ME, replace=True)
    # while new_alphas == alphas:
    #     np.random.seed()
    #     new_alphas = np.random.choice(alphas, ME, replace=True)



    np.random.seed(seed)
    if experiment_id == 1:
        ME_concept_drift_rounds = [[0.3], [0.6]]
        new_alphas = [[alphas[1]], [alphas[0]]]

    return ME_concept_drift_rounds, new_alphas




