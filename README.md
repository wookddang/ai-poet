# 시스템 아키텍처 (VMware 기반 Rocky Linux 9.0 클러스터)

```mermaid
flowchart TD
    subgraph Building1["🏢 Building 1 (Edge)"]
        A1[Server Node]
        A2[(Smart Meter & Sensors)]
        A1 -->|Collector Pod| C1[Docker Container: Collector]
        A1 -->|Analyzer Pod| C2[Docker Container: AI Analyzer]
        A1 -->|Exporter Pod| C3[Docker Container: Metrics Exporter]
    end

    subgraph Building2["🏢 Building 2 (Edge)"]
        B1[Server Node]
        B2[(Smart Meter & Sensors)]
        B1 -->|Collector Pod| D1[Docker Container: Collector]
        B1 -->|Analyzer Pod| D2[Docker Container: AI Analyzer]
        B1 -->|Exporter Pod| D3[Docker Container: Metrics Exporter]
    end

    subgraph CentralCluster["☸️ Kubernetes Cluster (Central)"]
        E1[MariaDB StatefulSet]
        E2[WordPress Deployment]
        E3[Grafana/PowerBI Connector]
    end

    subgraph Automation["⚙️ Ansible Control Node"]
        F1[Playbook: Install Docker & K8s]
        F2[Playbook: Deploy Collector Modules]
        F3[Playbook: Configure K8s Namespace]
    end

    A2 --> C1
    B2 --> D1

    C1 --> E1
    D1 --> E1
    C2 --> E1
    D2 --> E1

    E1 --> E2
    E1 --> E3

    F1 --> A1
    F1 --> B1
    F2 --> A1
    F2 --> B1
    F3 --> CentralCluster
