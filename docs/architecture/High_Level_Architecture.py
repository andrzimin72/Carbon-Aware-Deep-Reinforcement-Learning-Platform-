#!/usr/bin/env python3
"""
Generates a High-Level Architecture PNG for the Carbon-Aware DRL Platform.
"""
from diagrams import Diagram, Cluster, Edge
from diagrams.saas import API
from diagrams.generic.database import Database
from diagrams.programming.language import Python
from diagrams.k8s.compute import Pod
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.queue import Kafka # Used as a generic message queue/flow

graph_attr = {
    "fontsize": "45",
    "bgcolor": "white",
    "pad": "2.0",
}

with Diagram(
    "Carbon-Aware DRL Platform\nHigh-Level Architecture",
    show=False,
    filename="docs/whitepaper/figures/high_level_architecture",
    graph_attr=graph_attr,
    direction="TB"
):

    # --- External Systems ---
    watttime = API("WattTime API", "Provides MOER grid data")
    k8s_cluster = Pod("Kubernetes Cluster", "Runs scheduled workloads")

    # --- Layer 1: Data Ingestion ---
    with Cluster("Data Ingestion Layer"):
        updater = Python("Data Updater", "daily_update.py")
        cache = Database("Parquet Cache", "90+ days hourly MOER")
        
        watttime >> Edge(label="Fetches historical data (HTTPS)") >> updater
        updater >> Edge(label="Writes aligned data") >> cache

    # --- Layer 2: AI/ML Core ---
    with Cluster("AI/ML Core"):
        buffer = Python("Replay Buffer", "Loads Parquet data")
        env = Python("Gym Environment", "AdvancedReplayEnv")
        agent = Python("PPO Agent (LSTM)", "RecurrentActorCriticPPO")
        
        cache >> Edge(label="Reads time-series") >> buffer
        buffer >> Edge(label="Feeds episodes") >> env
        env >> Edge(label="Observation vectors") >> agent

    # --- Layer 3: Production Runtime ---
    with Cluster("Production Runtime"):
        controller = Python("K8s Controller", "k8s_controller.py")
        shadow = Python("Shadow Runner", "infer_shadow.py")
        breaker = Python("Circuit Breaker", "Dynamic risk management")
        
        agent >> Edge(label="Loads checkpoint (.pt)") >> controller
        agent >> Edge(label="Loads checkpoint (.pt)") >> shadow
        controller >> Edge(label="Binds Pods (API)") >> k8s_cluster
        controller >> Edge(label="Checks stability") >> breaker
        shadow >> Edge(label="Validates before prod") >> controller

    # --- Layer 4: Observability ---
    with Cluster("Observability"):
        prom = Prometheus("Prometheus", "Metrics collection")
        graf = Grafana("Grafana", "Dashboards")
        logs = Database("Audit Logs", "shadow_decisions.jsonl")
        
        controller >> Edge(label="Exposes :8000/metrics") >> prom
        shadow >> Edge(label="Exposes :8000/metrics") >> prom
        prom >> Edge(label="Scrapes metrics") >> graf
        controller >> Edge(label="Writes decisions") >> logs
        shadow >> Edge(label="Writes decisions") >> logs

print("✅ Architecture diagram generated successfully!")
print("📁 Saved to: docs/whitepaper/figures/high_level_architecture.png")
