import subprocess

# Experiment 1
executions = [
              #   "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
              # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
              # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
              # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
              # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1_new_clients'",
              # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1_new_clients'",
              # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1'",
              # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='CIFAR10' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1'",
              # "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1'",
              "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='EMNIST' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1'",
              "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=0.1 --experiment_id='1'",
              "python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy='FedYogi+FP' --dataset='GTSRB' --model='CNN_3' --fraction_fit=0.3 --alpha=1.0 --experiment_id='1'",
              ]

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

for i, execution in enumerate(executions):
    print(f"Execution {i+1}")
    subprocess.Popen(execution, shell=True).wait()