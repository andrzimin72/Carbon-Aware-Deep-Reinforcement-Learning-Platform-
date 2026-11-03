# Prerequisites Guide

This document specifies the hardware and software requirements for deploying the Carbon-Aware DRL Platform in development, staging, and production environments.

---

## Table of Contents

1. [Hardware Specifications]
   - [Development Environment]
   - [Staging Environment]
   - [Production Environment]
   - [GPU Requirements (Optional)]
   - [Storage Requirements]
   - [Network Requirements]
2. [Software Stack]
   - [Operating Systems]
   - [Python Environment]
   - [Kubernetes]
   - [Container Runtime]
   - [Monitoring Stack]
   - [External Dependencies]
3. Installation Verification
   - [Pre-Flight Checklist]
   - [Quick Start Verification]
---

## 1. Hardware Specifications

### 1.1 Development Environment

**Purpose:** Local development, testing, and small-scale experiments.

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| **CPU** | 4 cores | 8 cores | Training is CPU-intensive |
| **RAM** | 8 GB | 16 GB | LSTM training requires memory |
| **Storage** | 50 GB SSD | 100 GB SSD | For data cache, logs, checkpoints |
| **Network** | 10 Mbps | 100 Mbps | WattTime API calls |

**Typical Hardware:**
- Laptop: MacBook Pro M1/M2 (16 GB RAM) or equivalent
- Workstation: Intel i7/Ryzen 7 with 16 GB RAM
- Cloud VM: AWS t3.xlarge (4 vCPU, 16 GB RAM)

**Use Cases:**
- Running unit tests
- Training on synthetic data
- Shadow mode with demo tasks
- Local Kubernetes (minikube, kind)

### 1.2 Staging Environment

**Purpose:** Pre-production validation, integration testing, shadow mode with real workloads.

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| **CPU** | 8 cores | 16 cores | Parallel training across seeds |
| **RAM** | 16 GB | 32 GB | Multiple training runs |
| **Storage** | 200 GB SSD | 500 GB SSD | 90+ days of data, multiple checkpoints |
| **Network** | 100 Mbps | 1 Gbps | Faster data downloads |

**Typical Hardware:**
- Cloud VM: AWS m5.2xlarge (8 vCPU, 32 GB RAM)
- On-premise: Dell PowerEdge R740 or equivalent
- Kubernetes cluster: 3 nodes (4 vCPU, 16 GB RAM each)

**Use Cases:**
- Training on real WattTime data
- Multi-seed ablation studies
- Shadow mode with production-like workloads
- Integration testing with Kubernetes

### 1.3 Production Environment

**Purpose:** Live scheduling of real workloads with high availability.

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| **CPU** | 4 cores (controller) | 8 cores (controller) | Inference is lightweight |
| **RAM** | 8 GB (controller) | 16 GB (controller) | LSTM inference + buffering |
| **Storage** | 100 GB SSD | 500 GB SSD | Logs, metrics, audit trail |
| **Network** | 1 Gbps | 10 Gbps | Low-latency Pod binding |
| **Availability** | Single node | 3-node cluster | High availability |

**Kubernetes Cluster Requirements:**

| Node Role | Count | CPU | RAM | Storage | Purpose |
|---|---|---|---|---|---|
| **Controller** | 1-3 | 4 cores | 8 GB | 100 GB | Runs the scheduler |
| **Worker** | 3+ | 8+ cores | 32+ GB | 500+ GB | Runs scheduled workloads |
| **Monitoring** | 1 | 2 cores | 4 GB | 200 GB | Prometheus, Grafana |

**Typical Hardware:**
- **Small deployment** (100-500 Pods/day):
  - Controller: AWS m5.xlarge (4 vCPU, 16 GB RAM)
  - Workers: 3× m5.2xlarge (8 vCPU, 32 GB RAM)
  
- **Medium deployment** (500-2000 Pods/day):
  - Controller: AWS m5.2xlarge (8 vCPU, 32 GB RAM)
  - Workers: 5-10× m5.4xlarge (16 vCPU, 64 GB RAM)
  
- **Large deployment** (2000+ Pods/day):
  - Controller: AWS m5.4xlarge (16 vCPU, 64 GB RAM)
  - Workers: 10-50× m5.8xlarge (32 vCPU, 128 GB RAM)
  - Consider auto-scaling for workers

**High Availability Considerations:**
- Controller: Use leader election (only one active instance)
- Data: Replicate WattTime cache across nodes
- Monitoring: Deploy Prometheus/Grafana on separate nodes
- Backups: Daily snapshots of checkpoints and logs

### 1.4 GPU Requirements (Optional)

**Purpose:** Accelerate training (not required for inference).

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| **GPU** | NVIDIA T4 (16 GB) | NVIDIA A100 (40 GB) | For large-scale training |
| **VRAM** | 8 GB | 16+ GB | LSTM + PPO memory requirements |
| **CUDA** | 11.8 | 12.1 | Must match PyTorch version |

**When to Use GPUs:**
- ✅ Training on large datasets (180+ days)
- ✅ Running many seeds in parallel
- ✅ Experimenting with larger models
- ❌ Inference (CPU is sufficient)
- ❌ Small datasets (< 90 days)

**GPU Cloud Instances:**
- AWS: g4dn.xlarge (T4), p3.2xlarge (V100), p4d.24xlarge (A100)
- GCP: n1-standard-8 + T4, a2-highgpu-1g (A100)
- Azure: Standard_NC6s_v3 (V100), Standard_ND96asr_v4 (A100)

**Cost Optimization:**
- Use spot/preemptible instances for training (70-90% cost savings)
- Train offline, deploy CPU-only for inference
- Schedule training during off-peak hours

### 1.5 Storage Requirements

**Breakdown by Component:**

| Component | Size | Growth Rate | Retention | Notes |
|---|---|---|---|---|
| **WattTime Cache** | ~10 MB/region/month | 10 MB/month | 1 year | Parquet files |
| **Checkpoints** | ~50 MB each | 50 MB/training | 5 versions | PyTorch models |
| **Training Logs** | ~100 MB/run | 100 MB/training | 30 days | JSON/CSV |
| **Shadow Decisions** | ~50 MB/day | 50 MB/day | 90 days | JSONL audit trail |
| **Controller Logs** | ~20 MB/day | 20 MB/day | 30 days | Rotated logs |
| **Prometheus Metrics** | ~500 MB/month | 500 MB/month | 30 days | Time-series DB |
| **Grafana Data** | ~10 MB | Minimal | Indefinite | Dashboard configs |

**Total Storage Estimates:**

| Environment | 30 Days | 90 Days | 1 Year |
|---|---|---|---|
| **Development** | 5 GB | 10 GB | 30 GB |
| **Staging** | 20 GB | 50 GB | 150 GB |
| **Production** | 50 GB | 150 GB | 500 GB |

**Storage Recommendations:**
- Use SSD for all environments (faster I/O for training)
- Enable log rotation (configured in `deploy/logrotate/carbon-aware`)
- Archive old checkpoints to object storage (S3, GCS)
- Monitor disk usage with Prometheus node-exporter

### 1.6 Network Requirements

**Inbound Connections:**

| Source | Port | Protocol | Purpose |
|---|---|---|---|
| Kubernetes API | 6443 | HTTPS | Pod binding requests |
| Prometheus | 9090 | HTTP | Metrics scraping |
| Grafana | 3000 | HTTP | Dashboard access |
| Controller Health | 8080 | HTTP | Liveness probes |
| Controller Metrics | 8000 | HTTP | Prometheus metrics |

**Outbound Connections:**

| Destination | Port | Protocol | Purpose |
|---|---|---|---|
| WattTime API | 443 | HTTPS | Fetch MOER data |
| Prometheus Pushgateway | 9091 | HTTP | (Optional) Push metrics |
| Slack/Discord Webhook | 443 | HTTPS | (Optional) Notifications |
| Kubernetes API | 6443 | HTTPS | Pod binding |

**Bandwidth Requirements:**

| Operation | Bandwidth | Frequency | Notes |
|---|---|---|---|
| WattTime API calls | ~1 KB/request | Every 5 min | Low bandwidth |
| Data download (initial) | ~10 MB/region | One-time | 90 days of data |
| Metrics export | ~10 KB/scrape | Every 15 sec | Prometheus |
| Log shipping | ~1 MB/hour | Continuous | To log aggregator |

**Latency Requirements:**
- WattTime API: < 2 seconds (acceptable)
- Kubernetes API: < 100 ms (critical for Pod binding)
- Prometheus scrape: < 500 ms (monitoring)

**Firewall Rules:**
bash
Allow outbound HTTPS to WattTime API
iptables -A OUTPUT -p tcp --dport 443 -d api.watttime.org -j ACCEPT
Allow inbound from Kubernetes API server
iptables -A INPUT -p tcp --dport 6443 -s <K8S_API_CIDR> -j ACCEPT
Allow inbound Prometheus scraping
iptables -A INPUT -p tcp --dport 8000 -s <PROMETHEUS_CIDR> -j ACCEPT

---

## 2. Software Stack

### 2.1 Operating Systems

**Supported Operating Systems:**

| OS | Version | Status | Notes |
|---|---|---|---|
| **Ubuntu** | 20.04 LTS, 22.04 LTS | ✅ Fully supported | Recommended for production |
| **Debian** | 11 (Bullseye), 12 (Bookworm) | ✅ Fully supported | Stable, minimal |
| **CentOS/RHEL** | 8, 9 | ✅ Supported | Enterprise environments |
| **Fedora** | 38+ | ⚠️ Tested | Rapid updates, may break |
| **macOS** | 12+ (Monterey) | ⚠️ Development only | Not for production |
| **Windows** | 10/11 | ❌ Not supported | Use WSL2 or Docker |

**Kernel Requirements:**
- Linux kernel 5.4+ (for cgroups v2 support)
- Recommended: 5.15+ (better container performance)

**System Packages (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y \
  python3.11 \
  python3.11-venv \
  python3-pip \
  curl \
  wget \
  git \
  jq \
  htop \
  iotop \
  net-tools
```

### 2.2 Python Environment
Python Version:
Required: Python 3.10 or 3.11
Recommended: Python 3.11 (best performance)
Not supported: Python 3.9 or earlier (missing type union syntax)
Virtual Environment:
```
# Create virtual environment
python3.11 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

Installation:
```
# Production dependencies
pip install -r requirements.txt

# Development dependencies (testing, linting)
pip install -r requirements-dev.txt

# GPU support (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Dependency Conflicts:
numpy must be < 2.0 (PyTorch compatibility)
torch version must match CUDA version
kubernetes client version should match K8s cluster version

### 2.3 Kubernetes
Kubernetes Version:
Required: Kubernetes 1.26+
Recommended: Kubernetes 1.28 LTS
Tested up to: Kubernetes 1.29

Required Kubernetes Features:
Custom Resource Definitions (CRDs) — not used, but available
Role-Based Access Control (RBAC) — required
Pod Security Policies (PSP) or Pod Security Admission (PSA) — recommended
Network Policies — optional, for security

Kubernetes Client Configuration:
```
# Verify kubectl is configured
kubectl cluster-info

# Check version
kubectl version --client

# Verify namespace exists
kubectl get namespace carbon-aware
```

RBAC Requirements:
The controller requires a ServiceAccount with specific permissions (see deploy/kubernetes/rbac.yaml):
pods: get, list, watch
pods/binding: create
nodes: get, list, watch
events: create, patch
configmaps: get, list, watch
secrets: get (for WattTime credentials)

### 2.4 Container Runtime

a) Docker Version:
Required: Docker 20.10+
Recommended: Docker 24.0+
Alternative: containerd 1.6+ (Kubernetes native)

b) Docker Compose:
Required: Docker Compose v2.0+
Installation: Included with Docker Desktop or install separately

Image Registry:
```
# Build images
docker build -f deploy/docker/Dockerfile -t carbon-aware-controller:latest .
docker build -f deploy/docker/Dockerfile.update -t carbon-aware-updater:latest .

# Push to registry
docker tag carbon-aware-controller:latest your-registry.io/carbon-aware-controller:v1.0.0
docker push your-registry.io/carbon-aware-controller:v1.0.0
```

Image Security:
All images run as non-root user (UID 1000)
Read-only root filesystem where possible
No unnecessary packages installed
Regular vulnerability scanning with Trivy or Snyk

### 2.5 Monitoring Stack

a) Prometheus:
Version: 2.48.0+
Purpose: Metrics collection and storage
Retention: 30 days (configurable)
Storage: ~500 MB/month

b) Grafana:
Version: 10.2.0+
Purpose: Dashboard visualization
Data source: Prometheus
Dashboards: Auto-provisioned from deploy/grafana/dashboards/

c) Alertmanager (Optional):
Version: 0.26.0+
Purpose: Alert routing and notification
Integrations: Slack, email, PagerDuty

d) Node Exporter (Optional):
Version: 1.6.0+
Purpose: Host-level metrics (CPU, memory, disk)
Deployment: DaemonSet on all nodes

e) Kepler (Optional):
Version: 0.7.0+
Purpose: Container-level energy metrics
Integration: Provides node utilization to scheduler

### 2.6 External Dependencies

a) WattTime API:
URL: https://api.watttime.org/v3
Authentication: JWT token (username/password)
Rate limit: 30 requests/minute (free tier)
Signals: MOER (Marginal Operating Emissions Rate)
Regions: U.S. balancing authorities (CAISO, MISO, ERCOT, etc.)
Documentation: https://www.watttime.org/api-documentation/

b) Alternative Data Sources (Experimental):
EIA API: U.S. Energy Information Administration (average CI, not MOER)
Electricity Maps: Global carbon intensity data (commercial)
Custom: Internal grid telemetry (requires custom integration)

## 3. Installation Verification
### 3.1 Pre-Flight Checklist
Run this script to verify all prerequisites are met:
```
#!/bin/bash
# verify_prerequisites.sh

set -e

echo "=========================================="
echo "Carbon-Aware Platform — Prerequisites Check"
echo "=========================================="

# Check Python version
echo -n "Python version: "
python3 --version || { echo "❌ Python 3 not found"; exit 1; }
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" < "3.10" ]]; then
  echo "❌ Python 3.10+ required (found $PYTHON_VERSION)"
  exit 1
fi
echo "✅ Python $PYTHON_VERSION"

# Check pip
echo -n "pip version: "
pip3 --version || { echo "❌ pip not found"; exit 1; }
echo "✅ pip installed"

# Check Docker
echo -n "Docker version: "
docker --version || { echo "❌ Docker not found"; exit 1; }
echo "✅ Docker installed"

# Check Docker Compose
echo -n "Docker Compose version: "
docker-compose --version || docker compose version || { echo "❌ Docker Compose not found"; exit 1; }
echo "✅ Docker Compose installed"

# Check kubectl
echo -n "kubectl version: "
kubectl version --client --short || { echo "❌ kubectl not found"; exit 1; }
echo "✅ kubectl installed"

# Check Kubernetes cluster
echo -n "Kubernetes cluster: "
kubectl cluster-info >/dev/null 2>&1 || { echo "❌ Kubernetes cluster not accessible"; exit 1; }
echo "✅ Kubernetes cluster accessible"

# Check WattTime credentials
echo -n "WattTime credentials: "
if [ -f .env ]; then
  source .env
  if [ -n "$WATTTIME_USERNAME" ] && [ -n "$WATTTIME_PASSWORD" ]; then
    echo "✅ Found in .env"
  else
    echo "❌ Missing in .env"
    exit 1
  fi
else
  echo "⚠️  .env file not found (will prompt during deployment)"
fi

# Check disk space
echo -n "Available disk space: "
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 50 ]; then
  echo "❌ Less than 50 GB available ($AVAILABLE_GB GB)"
  exit 1
fi
echo "✅ $AVAILABLE_GB GB available"

# Check memory
echo -n "Available memory: "
AVAILABLE_GB=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAILABLE_GB" -lt 8 ]; then
  echo "⚠️  Less than 8 GB available ($AVAILABLE_GB GB)"
else
  echo "✅ $AVAILABLE_GB GB available"
fi

echo ""
echo "=========================================="
echo "✅ All prerequisites verified!"
echo "=========================================="
```

### 3.2 Quick Start Verification
```
#!/bin/bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run unit tests
pytest tests/ -v

# 3. Generate mock data (if WattTime credentials not available)
python scripts/generate_mock_cache.py

# 4. Train a test model (5 episodes, quick validation)
python -m src.training.replay_trainer \
  --window-size 5 \
  --seed 42 \
  --episodes 5 \
  --checkpoint-path /tmp/test.pt

# 5. Run shadow mode (10 decisions)
python -m src.deployment.infer_shadow \
  --checkpoint /tmp/test.pt \
  --regions CAISO_NORTH MISO ERCOT \
  --interval 1 \
  --max-decisions 10

# 6. Verify metrics endpoint
curl http://localhost:8000/metrics | grep carbon_aware

# 7. Deploy to Kubernetes (dry run)
kubectl apply -k deploy/kubernetes/ --dry-run=client

echo "✅ All verification steps passed!"
```
