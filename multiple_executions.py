import subprocess

# Experiment 1
# executions = [
#                 "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1'",
#               ]

# Experiment 1 new clients
# executions = [
#                 "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'", # este
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
#               "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
#               ]

# Experiment 1 CNN_2
# executions = [
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls_compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='fedkd'",
#             "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='sparsification'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='per'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls_compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='fedkd'",
#             "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='sparsification'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='per'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls_compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='fedkd'",
#             "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='sparsification'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='per'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls_compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='fedkd'",
#             "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='sparsification'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='per'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls_compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='fedkd'",
#             "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='sparsification'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='per'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls_compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='fedkd'",
#             "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='sparsification'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='per'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls_compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='compredict'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='fedkd'",
#             "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='sparsification'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='fedkd'",
#             # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_2' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='per'",
#               ]

# Experiment 1 CNN_3
executions = [
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='fedkd'",
            "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='sparsification'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls_compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='per'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls_compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='fedkd'",
            "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='sparsification'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='per'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls_compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='fedkd'",
            "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='sparsification'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='per'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls_compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='fedkd'",
            "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='sparsification'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='per'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls_compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='fedkd'",
            "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='sparsification'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='per'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='dls'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='fedkd'",
            "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='sparsification'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression='per'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='dls_compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='compredict'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='fedkd'",
            "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='sparsification'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='fedkd'",
            # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression='per'",
              ]

# executions = [
#
#     # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#     # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.5 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.5 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.7 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.7 --alpha=1.0 --experiment_id='1' --compression=''",
#
#     # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#     # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.5 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.5 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.7 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvg' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.7 --alpha=1.0 --experiment_id='1' --compression=''",
#
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.5 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.5 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.7 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.7 --alpha=1.0 --experiment_id='1' --compression=''",
#
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.5 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.5 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.7 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.7 --alpha=1.0 --experiment_id='1' --compression=''",
#
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.5 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.5 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.7 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.7 --alpha=1.0 --experiment_id='1' --compression=''",
#
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.5 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.5 --alpha=1.0 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.7 --alpha=0.1 --experiment_id='1' --compression=''",
#     "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedAvgPOC+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.7 --alpha=1.0 --experiment_id='1' --compression=''",
# ]

# Nome do arquivo
filename = "execution_process.txt"

# Etapa 1: Limpa (ou cria) o arquivo
with open(filename, "w") as f:
    pass  # Apenas limpa o conteúdo (ou cria se não existir)

for i, execution in enumerate(executions):
    execution_id = i+1
    print(f"Execution {execution_id}")
    with open(filename, "a") as f:
        f.write(f"Execution {execution_id}\n{execution}\n")

    subprocess.Popen(execution, shell=True).wait()