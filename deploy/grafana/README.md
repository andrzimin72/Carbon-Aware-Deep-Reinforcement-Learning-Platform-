## Grafana Dashboards for the Carbon-Aware Deep Reinforcement Learning Platform

Real-time observability for carbon-aware scheduling decisions, grid carbon intensity,
agent behavior, and risk management state.

### 📊 Dashboard Overview

The dashboard is organized into four logical sections:

#### 1. System Overview
- **System Health**: Binary indicator of overall runner health
- **Circuit Breaker**: Shows whether risk management has degraded to safe mode
- **Total Decisions**: Cumulative count of scheduling decisions
- **Acceptance Rate**: Percentage of decisions accepted by safety gate (gauge)

#### 2. Risk & Stability
- **Decision & Error Rates**: Real-time rates of decisions vs. errors
- **WattTime API Error Rate**: Monitors upstream API health

#### 3. Carbon Impact
- **Grid Carbon Intensity**: Real-time MOER across all regions with threshold lines
  - Green zone (<250 gCO₂e/kWh): optimal for scheduling
  - Yellow zone (250-500): moderate
  - Red zone (>500): avoid if possible

#### 4. Agent Behavior
- **Policy Confidence**: How confident the LSTM is in its decisions
- **Critic Value**: State value estimate (detect model drift)
- **Cumulative Decision Outcomes**: Stacked view of accepted/overridden/errors
- **Override Rate**: Safety gate intervention frequency

### 🚀 Quick Start

### Option A: Auto-Provisioning (Recommended)

The dashboard is automatically loaded when Grafana starts via the docker-compose stack:

```bash
cd deploy/docker
docker-compose up -d
# Dashboard available at http://localhost:3000/d/carbon-aware-swarm-platform
