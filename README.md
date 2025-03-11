---
title: Leveraging Flower and Docker for Device Heterogeneity Management in FL
tags: [deployment, vision, tutorial]
dataset: [CIFAR-10]
framework: [Docker, torch]
---

# FL-H.IAAC

<p align="center">
  <img src="https://flower.ai/_next/image/?url=%2F_next%2Fstatic%2Fmedia%2Fflower_white_border.c2012e70.png&w=640&q=75" width="140px" alt="Flower Website" />
  <img src="https://github.com/ChoosyDevs/Choosy/assets/59146613/73d15990-453b-4da6-b8d6-df0f956a127c" width="140px" alt="Docker Logo" />
</p>

## Introduction

This project uses the Flower framework for Federated Learning (FL) simulation. It is executed via docker to enable realistic distributed simulation. 

### Highlights

FL-H.IAAC highlights are presented as follows:
- Reproductibility: run experiments with only one command.
- Data heterogeneity: simulate non-IID data across devices.
- System heterogeneity: simulate devices with heterogeneous resources.
- MEFL: support for Multi-model Federated Learning.
- GPU usage: it is configured to use GPU when available.

## Enabling containers to access GPU

Follow this tutorial (`https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`) to install nvidia-container-toolkit

## Handling Device Heterogeneity

1. **System Metrics Access**:

   - Effective management of device heterogeneity begins with monitoring system metrics of each container. We integrate the following services to achieve this:
     - **Cadvisor**: Collects comprehensive metrics from each Docker container.
     - **Prometheus**: Using `prometheus.yaml` for configuration, it scrapes data from Cadvisor at scheduled intervals, serving as a robust time-series database. Users can access the Prometheus UI at `http://localhost:9090` to create and run queries using PromQL, allowing for detailed insight into container performance.

2. **Mitigating Heterogeneity**:

   - In this basic use case, we address device heterogeneity by establishing rules tailored to each container's system capabilities. This involves modifying training parameters, such as batch sizes and learning rates, based on each device's memory capacity and CPU availability. These settings are specified in the `client_configs` array in the `create_docker_compose` script. For example:

     ```python
     client_configs = [
           {"mem_limit": "3g", "batch_size": 32, "cpus": 4, "learning_rate": 0.001},
           {"mem_limit": "6g", "batch_size": 256, "cpus": 1, "learning_rate": 0.05},
           {"mem_limit": "4g", "batch_size": 64, "cpus": 3, "learning_rate": 0.02},
           {"mem_limit": "5g", "batch_size": 128, "cpus": 2.5, "learning_rate": 0.09},
     ]
     ```

## Prerequisites

Docker must be installed and the Docker daemon running on your server. If you don't already have Docker installed, you can get [installation instructions for your specific Linux distribution or macOS from Docker](https://docs.docker.com/engine/install/). Besides Docker, the only extra requirement is having Python installed. You don't need to create a new environment for this example since all dependencies will be installed inside Docker containers automatically.

## Running the traditional FL Example

Run the training with a single command:

```bash

# Generate docker compose file and automatically run containers
python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=100 --strategy="FedAvg" --dataset="CIFAR10" --model="CNN_3" --fraction_fit=0.3 --alpha=0.1
```

## Running the MEFL Example

In MEFL, the standard solution is the `MultiFedAvg`.
For each additional task, add the fields `dataset`, `model`, and `alpha`:

```bash

python helpers/generate_docker_compose.py --total_clients=20 --number_of_rounds=10 --strategy="MultiFedAvg" --dataset="CIFAR10" --dataset="EMNIST" --model="CNN" --model="CNN" --fraction_fit=0.3 --alpha=0.1 --alpha=0.1

```

## Saving the results locally:

Save the results/ volume locally:

```bash

bash get_results.sh
```

Once you finish your simulation, type the following command:

```bash

docker compose down
```

In case your building is not working, try the following steps (one at a time):
```bash
# Remove unused containers
docker container prune -f

```

When changing strategy it is important to first remove existing images:

```bash
# Remove images
docker image prune -a

```

```bash
# Remove everything (images, containers, networks)
docker system prune -a --volumes

```

On your favourite browser, go to `http://localhost:3000` to see the Graphana dashboard showing system-level and application-level metrics.

To stop all containers, open a new terminal and `cd` into this directory, then run `docker-compose down`. Alternatively, you can do `ctrl+c` on the same terminal and then run `docker-compose down` to ensure everything is terminated.

## Model Training and Dataset Integration

### Data Pipeline with FLWR-Datasets

We have integrated [`flwr-datasets`](https://flower.ai/docs/datasets/) into our data pipeline, which is managed within the `load_data.py` file in the `helpers/` directory. This script facilitates standardized access to datasets across the federated network and incorporates a `data_sampling_percentage` argument. This argument allows users to specify the percentage of the dataset to be used for training and evaluation, accommodating devices with lower memory capabilities to prevent Out-of-Memory (OOM) errors.

### Model Selection and Dataset

For the federated learning system, we have selected the MobileNet model due to its efficiency in image classification tasks. The model is trained and evaluated on the CIFAR-10 dataset. The combination of MobileNet and CIFAR-10 is ideal for demonstrating the capabilities of our federated learning solution in a heterogeneous device environment.

- **MobileNet**: A streamlined architecture for mobile and embedded devices that balances performance and computational cost.
- **CIFAR-10 Dataset**: A standard benchmark dataset for image classification, containing various object classes that pose a comprehensive challenge for the learning model.

By integrating these components, our framework is well-prepared to handle the intricacies of training over a distributed network with varying device capabilities and data availability.

## Visualizing with Grafana

### Access Grafana Dashboard

Visit `http://localhost:3000` to enter Grafana. The automated setup ensures that you're greeted with a series of pre-configured dashboards, including the default screen with a comprehensive set of graphs. These dashboards are ready for immediate monitoring and can be customized to suit your specific requirements.

### Dashboard Configuration

The `dashboard_index.json` file, located in the `./config/provisioning/dashboards` directory, serves as the backbone of our Grafana dashboard's configuration. It defines the structure and settings of the dashboard panels, which are rendered when you access Grafana. This JSON file contains the specifications for various panels such as model accuracy, CPU usage, memory utilization, and network traffic. Each panel's configuration includes the data source, queries, visualization type, and other display settings like thresholds and colors.

For instance, in our project setup, the `dashboard_index.json` configures a panel to display the model's accuracy over time using a time-series graph, and another panel to show the CPU usage across clients using a graph that plots data points as they are received. This file is fundamental for creating a customized and informative dashboard that provides a snapshot of the federated learning system's health and performance metrics.

By modifying the `dashboard_index.json` file, users can tailor the Grafana dashboard to include additional metrics or change the appearance and behavior of existing panels to better fit their monitoring requirements.

### Grafana Default Dashboard

Below is the default Grafana dashboard that users will see upon accessing Grafana:

<img width="1440" alt="grafana_home_screen" src="https://github.com/ChoosyDevs/Choosy/assets/59146613/46c1016d-2376-4fdc-ae5f-68c550fc8e46">

This comprehensive dashboard provides insights into various system metrics across client-server containers. It includes visualizations such as:

- **Application Metrics**: The "Model Accuracy" graph shows an upward trend as rounds of training progress, which is a positive indicator of the model learning and improving over time. Conversely, the "Model Loss" graph trends downward, suggesting that the model is becoming more precise and making fewer mistakes as it trains.

- **CPU Usage**: The sharp spikes in the red graph, representing "client1", indicate peak CPU usage, which is considerably higher than that of "client2" (blue graph). This difference is due to "client1" being allocated more computing resources (up to 4 CPU cores) compared to "client2", which is limited to just 1 CPU core, hence the more subdued CPU usage pattern.

- **Memory Utilization**: Both clients are allocated a similar amount of memory, reflected in the nearly same lines for memory usage. This uniform allocation allows for a straightforward comparison of how each client manages memory under similar conditions.

- **Network Traffic**: Monitor incoming and outgoing network traffic to each client, which is crucial for understanding data exchange volumes during federated learning cycles.

Together, these metrics paint a detailed picture of the federated learning operation, showcasing resource usage and model performance. Such insights are invaluable for system optimization, ensuring balanced load distribution and efficient model training.

## Comprehensive Monitoring System Integration

### Capturing Container Metrics with cAdvisor

cAdvisor is seamlessly integrated into our monitoring setup to capture a variety of system and container metrics, such as CPU, memory, and network usage. These metrics are vital for analyzing the performance and resource consumption of the containers in the federated learning environment.

### Custom Metrics: Setup and Monitoring via Prometheus

In addition to the standard metrics captured by cAdvisor, we have implemented a process to track custom metrics like model's accuracy and loss within Grafana, using Prometheus as the backbone for metric collection.

1. **Prometheus Client Installation**:

   - We began by installing the `prometheus_client` library in our Python environment, enabling us to define and expose custom metrics that Prometheus can scrape.

2. **Defining Metrics in Server Script**:

   - Within our `server.py` script, we have established two key Prometheus Gauge metrics, specifically tailored for monitoring our federated learning model: `model_accuracy` and `model_loss`. These custom gauges are instrumental in capturing the most recent values of the model's accuracy and loss, which are essential metrics for evaluating the model's performance. The gauges are defined as follows:

     ```python
     from prometheus_client import Gauge

     accuracy_gauge = Gauge('model_accuracy', 'Current accuracy of the global utils')
     loss_gauge = Gauge('model_loss', 'Current loss of the global utils')
     ```

3. **Exposing Metrics via HTTP Endpoint**:

   - We leveraged the `start_http_server` function from the `prometheus_client` library to launch an HTTP server on port 8000. This server provides the `/metrics` endpoint, where the custom metrics are accessible for Prometheus scraping. The function is called at the end of the `main` method in `server.py`:

     ```python
     start_http_server(8000)
     ```

4. **Updating Metrics Recording Strategy**:

   - The core of our metrics tracking lies in the `strategy.py` file, particularly within the `aggregate_evaluate` method. This method is crucial as it's where the federated learning model's accuracy and loss values are computed after each round of training with the aggregated data from all clients.

     ```python
        self.accuracy_gauge.set(accuracy_aggregated)
        self.loss_gauge.set(loss_aggregated)
     ```

5. **Configuring Prometheus Scraping**:

   - In the `prometheus.yml` file, under `scrape_configs`, we configured a new job to scrape the custom metrics from the HTTP server. This setup includes the job's name, the scraping interval, and the target server's URL.

### Visualizing the Monitoring Architecture

The image below depicts the Prometheus scraping process as it is configured in our monitoring setup. Within this architecture:

- The "Prometheus server" is the central component that retrieves and stores metrics.
- "cAdvisor" and the "HTTP server" we set up to expose our custom metrics are represented as "Prometheus targets" in the diagram. cAdvisor captures container metrics, while the HTTP server serves our custom `model_accuracy` and `model_loss` metrics at the `/metrics` endpoint.
- These targets are periodically scraped by the Prometheus server, aggregating data from both system-level and custom performance metrics.
- The aggregated data is then made available to the "Prometheus web UI" and "Grafana," as shown, enabling detailed visualization and analysis through the Grafana dashboard.

<img width="791" alt="prometheus-architecture" src="https://github.com/ChoosyDevs/Choosy/assets/59146613/3b915e04-f12c-4aef-99ff-d75853234728">

By incorporating these steps, we have enriched our monitoring capabilities to not only include system-level metrics but also critical performance indicators of our federated learning model. This approach is pivotal for understanding and improving the learning process. Similarly, you can apply this methodology to track any other metric that you find interesting or relevant to your specific needs. This flexibility allows for a comprehensive and customized monitoring environment, tailored to the unique aspects and requirements of your federated learning system.

## Additional Resources

- **Grafana Tutorials**: Explore a variety of tutorials on Grafana at [Grafana Tutorials](https://grafana.com/tutorials/).
- **Prometheus Overview**: Learn more about Prometheus at their [official documentation](https://prometheus.io/docs/introduction/overview/).
- **cAdvisor Guide**: For information on monitoring Docker containers with cAdvisor, see this [Prometheus guide](https://prometheus.io/docs/guides/cadvisor/).

## Conclusion

This project serves as a foundational example of managing device heterogeneity within the federated learning context, employing the Flower framework alongside Docker, Prometheus, and Grafana. It's designed to be a starting point for users to explore and further adapt to the complexities of device heterogeneity in federated learning environments.
