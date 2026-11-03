#!/usr/bin/env python3
"""
Generates a high-resolution Gantt-style Strategic Roadmap.
Requires: matplotlib, numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==============================================================================
# 1. ROADMAP DATA
# Format: (Task Name, Start Month, Duration in Months, Color Hex)
# ==============================================================================
tasks = [
    # Phase 1: Foundation & Shadow Validation (Months 1-3)
    ("Phase 1: Foundation & Shadow Validation", 1, 3, "#2E86AB"),      # Deep Blue
    ("  M1.1: Infrastructure & Integration", 1, 1, "#5DADE2"),         # Light Blue
    ("  M1.2: Shadow Mode Validation", 2, 1, "#5DADE2"),
    ("  M1.3: Security & Compliance Gate", 3, 1, "#5DADE2"),
    
    # Phase 2: Controlled Production & ROI (Months 4-6)
    ("Phase 2: Controlled Production & ROI", 4, 3, "#1D8348"),         # Deep Green
    ("  M2.1: Batch Workload Enablement", 4, 1, "#58D68D"),            # Light Green
    ("  M2.2: Automated MLOps Pipeline", 5, 1, "#58D68D"),
    ("  M2.3: First ESG & Financial Reporting", 6, 1, "#58D68D"),
    
    # Phase 3: Enterprise Scaling & Advanced AI (Months 7-12)
    ("Phase 3: Enterprise Scaling & Advanced AI", 7, 6, "#B9770E"),    # Deep Orange
    ("  M3.1: Explicit Forecast Integration", 8, 1, "#F5B041"),        # Light Orange
    ("  M3.2: Multi-Objective RL (MORL)", 10, 1, "#F5B041"),
    ("  M3.3: Full Production & Multi-Region", 12, 1, "#F5B041"),
    
    # Phase 4: Innovation & Ecosystem Expansion (Months 13-18)
    ("Phase 4: Innovation & Ecosystem Expansion", 13, 6, "#6C3483"),   # Deep Purple
    ("  M4.1: Hardware-Aware Scheduling", 14, 1, "#AF7AC5"),           # Light Purple
    ("  M4.2: Energy Market Integration", 16, 1, "#AF7AC5"),
    ("  M4.3: Federated Learning & Open Ecosystem", 18, 1, "#AF7AC5"),
]

# ==============================================================================
# 2. PLOT CONFIGURATION
# ==============================================================================
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FAFAFA')

y_pos = np.arange(len(tasks))

# ==============================================================================
# 3. DRAWING THE GANTT CHART
# ==============================================================================
for i, (task, start, duration, color) in enumerate(tasks):
    # Draw the bar
    ax.barh(y_pos[i], duration, left=start, height=0.65, 
            color=color, edgecolor='#FFFFFF', linewidth=1.5, align='center', zorder=3)
    
    # Add text labels
    if "Phase" in task:
        # Center text inside the Phase bar
        ax.text(start + duration/2, y_pos[i], task, 
                ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=11, zorder=4)
    else:
        # Left-align text next to the Milestone bar
        ax.text(start + duration + 0.15, y_pos[i], task, 
                ha='left', va='center', color='#2C3E50', fontsize=10, fontweight='medium', zorder=4)

# ==============================================================================
# 4. FORMATTING & STYLING
# ==============================================================================
# Hide Y-axis ticks/labels (we use custom text)
ax.set_yticks(y_pos)
ax.set_yticklabels([""] * len(tasks))

# Title and Labels
ax.set_title("Carbon Aware Deep Reinforcement Learning Platform: 18-Month Strategic Roadmap", 
             fontsize=18, fontweight='bold', color='#1A1A1A', pad=25)
ax.set_xlabel("Project Timeline (Months)", fontsize=12, fontweight='bold', color='#34495E', labelpad=10)

# X-axis limits and grid
ax.set_xlim(0, 21) # Extra space for text labels on the right
ax.set_xticks(range(1, 19))
ax.set_xticklabels([f"Month {i}" for i in range(1, 19)], rotation=45, ha='right', fontsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.5, color='#BDC3C7', zorder=1)

# Clean up spines (borders)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#BDC3C7')

# Legend
legend_patches = [
    mpatches.Patch(color='#2E86AB', label='Phase 1: Foundation & Shadow Validation (M1-M3)'),
    mpatches.Patch(color='#1D8348', label='Phase 2: Controlled Production & ROI (M4-M6)'),
    mpatches.Patch(color='#B9770E', label='Phase 3: Enterprise Scaling & Advanced AI (M7-M12)'),
    mpatches.Patch(color='#6C3483', label='Phase 4: Innovation & Ecosystem (M13-M18)'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=11, frameon=True, 
          edgecolor='#BDC3C7', facecolor='#FFFFFF')

# Adjust layout to prevent clipping
plt.tight_layout()

# ==============================================================================
# 5. SAVE AS HIGH-RES PNG
# ==============================================================================
output_filename = "strategic_roadmap_gantt.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✅ Successfully generated: {output_filename}")
