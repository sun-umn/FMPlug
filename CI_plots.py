import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

# -------------------------
# 1. Load CSV
# -------------------------
csv_file = "performance_metrics_DIP-noiseless-CI.csv"  # replace with your path
df = pd.read_csv(csv_file)

# Metrics to analyze
# metrics = ['psnr', 'ssim', 'lpips', 'dists', 'clipiqa', 'musiq']
metrics = ['ssim', 'lpips']

# Create output folder for plots
os.makedirs("plots", exist_ok=True)

# -------------------------
# 2. Plot metrics vs difficulty with CI width on secondary axis
# -------------------------
def plot_metrics_vs_difficulty(df_task, task_name):
    # colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    colors = ['orange', 'green']
    plt.figure(figsize=(10,6))
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()  # secondary axis for CI width
    
    for m, c in zip(metrics, colors):
        mean_col = f"{m}_mean"
        ci_col = f"{m}_ci95"
        if mean_col in df_task.columns:
            # primary y-axis: metric mean ± CI
            ax1.errorbar(df_task['level'], df_task[mean_col],
                         yerr=df_task[ci_col], fmt='-o', color=c,
                         capsize=4, label=f"{m.upper()} mean")
            # secondary y-axis: CI width trend
            ax2.plot(df_task['level'], df_task[ci_col], '--', color=c, alpha=0.5, label=f"{m.upper()} CI width")
    
    ax1.set_xlabel('Difficulty level')
    ax1.set_ylabel('Metric mean')
    ax2.set_ylabel('95% CI width')
    ax1.set_title(f'{task_name}: Metrics vs Difficulty')
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=9)
    
    plt.grid(True)
    plt.tight_layout()
    # Save plot
    save_path = f"plots/{task_name}_metrics_vs_difficulty.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

# -------------------------
# 3. Compute Spearman correlation per metric
# -------------------------
def compute_trends(df_task):
    trend_results = []
    for m in metrics:
        mean_col = f"{m}_mean"
        if mean_col in df_task.columns:
            rho, pval = spearmanr(df_task['level'], df_task[mean_col])
            ci_col = f"{m}_ci95"
            rho_ci, pval_ci = spearmanr(df_task['level'], df_task[ci_col])
            trend_results.append({
                'metric': m.upper(),
                'rho_mean_vs_level': rho,
                'pval_mean_vs_level': pval,
                'rho_ci_vs_level': rho_ci,
                'pval_ci_vs_level': pval_ci
            })
    return pd.DataFrame(trend_results)

# -------------------------
# 4. Analyze all tasks
# -------------------------
tasks = df['task'].unique()
summary_all = []

for task_name in tasks:
    df_task = df[df['task'] == task_name].sort_values('level')
    print(f"\n--- Task: {task_name} ---")
    
    # Plot metrics with CI and CI width
    plot_metrics_vs_difficulty(df_task, task_name)
    
    # Compute trends
    trend_df = compute_trends(df_task)
    print(trend_df)
    summary_all.append((task_name, trend_df))

# -------------------------
# 5. Save summary table
# -------------------------
summary_output = pd.concat([df.assign(task=task) for task, df in summary_all], ignore_index=True)
summary_output.to_csv("metric_trends_summary.csv", index=False)
print("\nSummary saved to 'metric_trends_summary.csv'")
