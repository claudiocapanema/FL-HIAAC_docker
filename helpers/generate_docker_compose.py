import argparse
import random

parser = argparse.ArgumentParser(description="Generated Docker Compose")
parser.add_argument(
    "--total_clients", type=int, default=10, help="Total clients to spawn (default: 2)"
)
parser.add_argument(
    "--number_of_rounds", type=int, default=5, help="Number of FL rounds (default: 100)"
)
parser.add_argument(
    "--data_percentage",
    type=float,
    default=0.6,
    help="Portion of client data to use (default: 0.6)",
)
parser.add_argument(
    "--random", action="store_true", help="Randomize client configurations"
)

parser.add_argument(
    "--strategy", type=str, default='FedAvg+FP', help="Strategy to use (default: FedAvg)"
)
parser.add_argument(
    "--alpha", type=float, default=0.1, help="Dirichlet alpha"
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
    "--dataset", type=str, default="CIFAR10"
)
parser.add_argument(
    "--model", type=str, default="CNN"
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
    "--learning_rate", type=float, default=0.001
)


def create_docker_compose(args):
    # cpus is used to set the number of CPUs available to the container as a fraction of the total number of CPUs on the host machine.
    # mem_limit is used to set the memory limit for the container.
    client_configs = [
        {"mem_limit": "3g", "batch_size": 32, "cpus": 4, "learning_rate": 0.001} for i in range(args.total_clients)
        # Add or modify the configurations depending on your host machine
    ]

    strategy_name = args.strategy
    files_dict = {"FedAvg": {"client_file": """client_fedavg.py""".format(""), "server_file": """server_fedavg.py""".format(strategy_name)},
                   "FedAvg+FP": {"client_file": """client_fedavg_fedpredict.py""".format(""), "server_file": """server_fedavg.py""".format(strategy_name)}}
    client_file = files_dict[strategy_name]["client_file"]
    server_file = files_dict[strategy_name]["server_file"]

    general_config = f"--total_clients={args.total_clients} --number_of_rounds={args.number_of_rounds} --data_percentage={args.data_percentage} --strategy={strategy_name} --alpha={args.alpha} --round_new_clients={args.round_new_clients} --fraction_new_clients={args.fraction_new_clients} --model='{args.model}' --cd='{args.cd}' --fraction_fit={args.fraction_fit} --batch_size={args.batch_size} --learning_rate={args.learning_rate}"
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
    command: python {server_file} {general_config} 
    environment:
      FLASK_RUN_PORT: 6000
      DOCKER_HOST_IP: host.docker.internal
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
    # Add client services
    for i in range(1, args.total_clients + 1):
        if args.random:
            config = random.choice(client_configs)
        else:
            config = client_configs[(i - 1) % len(client_configs)]
        docker_compose_content += f"""
  client{i}:
    container_name: client{i}
    build:
      context: .
      dockerfile: Dockerfile
    command: python {client_file} --server_address=server:8080 --client_id={i} {general_config}
    deploy:
      resources:
        limits:
          cpus: "{(config['cpus'])}"
          memory: "{config['mem_limit']}"
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
    stop_signal: SIGINT
"""

    docker_compose_content += "volumes:\n  grafana-storage:\n"

    with open("docker-compose.yml", "w") as file:
        file.write(docker_compose_content)


if __name__ == "__main__":
    args = parser.parse_args()
    create_docker_compose(args)
