import argparse

parser = argparse.ArgumentParser(description="Generated Docker Compose")
parser.add_argument(
    "--total_clients", type=int, default=20, help="Total clients to spawn (default: 2)"
)
parser.add_argument(
    "--number_of_rounds", type=int, default=5, help="Number of FL rounds (default: 5)"
)
parser.add_argument(
    "--data_percentage",
    type=float,
    default=0.8,
    help="Portion of client data to use (default: 0.6)",
)
parser.add_argument(
    "--random", action="store_true", help="Randomize client configurations"
)

parser.add_argument(
    "--strategy", type=str, default='FedAvg', help="Strategy to use (default: FedAvg)"
)
parser.add_argument(
    "--alpha", action="append", help="Dirichlet alpha"
)
parser.add_argument(
    "--round_new_clients", type=float, default=0.1, help=""
)
parser.add_argument(
    "--fraction_new_clients", type=float, default=0.1, help=""
)
parser.add_argument(
    "--local_epochs", type=float, default=1, help=""
)
parser.add_argument(
    "--dataset", action="append"
)
parser.add_argument(
    "--model", action="append"
)
parser.add_argument(
    "--cd", type=str, default="false"
)
parser.add_argument(
    "--fraction_fit", type=float, default=0.3
)
parser.add_argument(
    "--client_id", type=int, default=1
)
parser.add_argument(
    "--batch_size", type=int, default=32
)
parser.add_argument(
    "--learning_rate", type=float, default=0.01
)

def assert_args(args, strategy_name):
    strategy_type = None
    if strategy_name in ["FedAvg", "FedAvg+FP", "FedYogi", "FedYogi+FP", "FedPer", "FedKD", "FedKD+FP"]:
        strategy_type = "FL"
    elif strategy_name in ["MultiFedAvg", "FedFairMMFL", "MultiFedEfficiency"]:
        strategy_type = "MEFL"

    args_size = True if len(args.dataset) == len(args.model) == len(args.alpha) else False
    if not args_size:
        raise Exception(f"Number of datasets and models and alpha should be the same but you gave: {len(args.dataset)} dataset(s) {len(args.model)} model(s) and {len(args.alpha)} alpha(s)")
    else:
        if len(args.dataset) == 1 and strategy_type == "MEFL":
            raise Exception(
                f"Strategy {strategy_name} is MEFL but you gave only {len(args.dataset)} dataset {len(args.model)} model and {len(args.alpha)} alpha"
            )
        elif len(args.dataset) > 1 and strategy_type == "FL":
            raise Exception(
                f"Strategy {strategy_name} is single model FL but you gave {len(args.dataset)} dataset(s) {len(args.model)} model(s) and {len(args.alpha)} alpha(s)"
            )


def create_docker_compose(args):
    assert_args(args, args.strategy)
    # cpus is used to set the number of CPUs available to the container as a fraction of the total number of CPUs on the host machine.
    # mem_limit is used to set the memory limit for the container.
    client_configs = [
        {"mem_limit": "1.2g", "cpus": 1} for i in range(args.total_clients)
        # Add or modify the configurations depending on your host machine
    ]

    strategy_name = args.strategy
    client_file = "start_client.py"
    server_file = "start_server.py"

    mefl_string = " "
    ME = len(args.dataset)
    for me  in range(ME):
        mefl_string += f" --dataset='{args.dataset[me]}' --model='{args.model[me]}' --alpha={float(args.alpha[me])} "


    general_config = f"--total_clients={args.total_clients} --number_of_rounds={args.number_of_rounds} --data_percentage={args.data_percentage} --strategy='{strategy_name}' --round_new_clients={args.round_new_clients} --fraction_new_clients={args.fraction_new_clients} --cd='{args.cd}' --fraction_fit={args.fraction_fit} --batch_size={args.batch_size} --learning_rate={args.learning_rate}" + mefl_string
    print("config geral: ", general_config)

    docker_compose_content = f"""
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - 9090:9090
    deploy:
      restart_policy:
        condition: on-failure
    command:
      - --config.file=/etc/prometheus/prometheus.yml
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - cadvisor

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    container_name: cadvisor
    privileged: true
    deploy:
      restart_policy:
        condition: on-failure
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
      - /var/run/docker.sock:/var/run/docker.sock

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - 3000:3000
    deploy:
      restart_policy:
        condition: on-failure
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./config/grafana.ini:/etc/grafana/grafana.ini
      - ./config/provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./config/provisioning/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
      - cadvisor
    command:
      - --config=/etc/grafana/grafana.ini

  server:
    container_name: server
    build:
      context: .
      dockerfile: Dockerfile
    runtime: nvidia
    command: python3 {server_file} {general_config}
    environment:
      FLASK_RUN_PORT: 6000
      DOCKER_HOST_IP: host.docker.internal
      NVIDIA_VISIBLE_DEVICES: all
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock
    ports:
      - "6000:6000"
      - "8265:8265"
      - "8000:8000"
    stop_signal: SIGINT
    depends_on:
      - prometheus
      - grafana
"""
    # client{i}:
    #     container_name: client{i}
    #     build:
    #       context: .
    #       dockerfile: Dockerfile
    #     command: python {client_file} --server_address=server:8080 --client_id={i} {general_config}
    #     deploy:
    #       resources:
    #         limits:
    #           cpus: "{(config['cpus'])}"
    #           memory: "{config['mem_limit']}"
    # Add client services
    # image: client{i}:latest
    for i in range(1, args.total_clients + 1):
        # if args.random:
        #     config = random.choice(client_configs)
        # else:
        #     config = client_configs[(i - 1) % len(client_configs)]
        config = ""
        docker_compose_content += f"""
  client{i}:
    container_name: client{i}
    build:
      context: .
      dockerfile: Dockerfile
    runtime: nvidia
    command: python3 {client_file} --server_address=server:8080 --client_id={i} {general_config}
    deploy:
          resources:
            limits:
              cpus: "{client_configs[0]['cpus']}"
              memory: "{client_configs[0]['mem_limit']}"
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock
    ports:
      - "{6000 + i}:{6000 + i}"
    depends_on:
      - server
    environment:
      FLASK_RUN_PORT: {6000 + i}
      container_name: client{i}
      DOCKER_HOST_IP: host.docker.internal
      NVIDIA_VISIBLE_DEVICES: all
    stop_signal: SIGINT
"""

    docker_compose_content += "volumes:\n  grafana-storage:\n"

    # filename = f"docker-compose_{strategy_name}_clients_{args.total_clients}_fraction_fit_{args.fraction_fit}_number_of_rounds_{args.number_of_rounds}_dataset_{args.dataset}_model_{args.utils}.yml"
    filename = f"docker-compose.yml"
    with open(filename, "w") as file:
        file.write(docker_compose_content)

    import subprocess

    # Caminho para o seu script bash
    script_up = f"sudo docker compose -f {filename} up --build && docker image prune -f"

    script_down = f"sudo docker compose -f {filename} down"
    subprocess.Popen(script_up, shell=True).wait()
    try:
        # Chamar o script bash usando subprocess
        subprocess.Popen(script_up, shell=True).wait()
        subprocess.Popen("sudo docker compose down", shell=True).wait()
        # subprocess.Popen("sudo bash get_results.sh", shell=True).wait()
    except Exception as e:
        # print(e)
        subprocess.Popen("sudo docker compose down", shell=True)
        pass
    print(script_down)




if __name__ == "__main__":
    args = parser.parse_args()
    create_docker_compose(args)
