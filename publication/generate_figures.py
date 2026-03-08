"""
Publication-ready figures for COVID-19 CDSS paper.
Generates 5 figures at 300 DPI.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1 — System Architecture Flowchart
# ---------------------------------------------------------------------------

def draw_box(ax, x, y, w, h, text, facecolor='#DBEAFE', edgecolor='#2563EB',
             fontsize=9, style='round,pad=0.1', linestyle='solid', zorder=3):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=style,
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.5, linestyle=linestyle, zorder=zorder)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=zorder+1, wrap=True,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, color='#374151'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8),
                zorder=4)

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # --- Left column (Mexican / Blue) ---
    # Input box
    draw_box(ax, 3.0, 7.8, 3.8, 0.9,
             'Patient Demographics\n& Comorbidities (17 features)',
             facecolor='#DBEAFE', edgecolor='#2563EB')
    draw_arrow(ax, 3.0, 7.35, 3.0, 6.65)

    # Model box
    draw_box(ax, 3.0, 6.2, 3.4, 0.8,
             'Mexican XGBoost\n(17 features)',
             facecolor='#BFDBFE', edgecolor='#1D4ED8')
    draw_arrow(ax, 3.0, 5.8, 3.0, 5.15)

    # Output box
    draw_box(ax, 3.0, 4.7, 3.4, 0.75,
             '5-class probabilities\n[Critical / Mild / Moderate\n/ No_COVID / Severe]',
             facecolor='#EFF6FF', edgecolor='#3B82F6', fontsize=8)

    # Left column label
    ax.text(3.0, 8.65, 'Always Available', ha='center', va='center',
            fontsize=9, color='#1D4ED8', fontstyle='italic')

    # --- Right column (Einstein / Green, dashed) ---
    draw_box(ax, 11.0, 7.8, 3.8, 0.9,
             'Lab Results (optional)\nCBC Panel (11 features)',
             facecolor='#DCFCE7', edgecolor='#16A34A', linestyle='dashed')
    draw_arrow(ax, 11.0, 7.35, 11.0, 6.65)

    draw_box(ax, 11.0, 6.2, 3.4, 0.8,
             'Einstein XGBoost\n(11 CBC features)',
             facecolor='#BBF7D0', edgecolor='#15803D', linestyle='dashed')
    draw_arrow(ax, 11.0, 5.8, 11.0, 5.15)

    draw_box(ax, 11.0, 4.7, 3.4, 0.75,
             '3-class probabilities\n[No_COVID / Non_Severe / Severe]\n→ mapped to 5 classes',
             facecolor='#F0FDF4', edgecolor='#22C55E', fontsize=8, linestyle='dashed')

    ax.text(11.0, 8.65, 'Lab values optional — system degrades gracefully',
            ha='center', va='center', fontsize=9, color='#15803D', fontstyle='italic')

    # --- Arrows from outputs to ensemble ---
    draw_arrow(ax, 3.0, 4.325, 7.0 - 1.3, 3.45)
    draw_arrow(ax, 11.0, 4.325, 7.0 + 1.3, 3.45)

    # --- Ensemble box (Purple) ---
    draw_box(ax, 7.0, 3.0, 4.2, 0.8,
             'Weighted Ensemble\n60% Mexican + 40% Einstein',
             facecolor='#F3E8FF', edgecolor='#7C3AED')
    draw_arrow(ax, 7.0, 2.6, 7.0, 1.95)

    # --- SHAP Explainer ---
    draw_box(ax, 7.0, 1.55, 3.0, 0.7,
             'SHAP Explainer',
             facecolor='#FEF3C7', edgecolor='#D97706', fontsize=9)
    draw_arrow(ax, 7.0, 1.2, 7.0, 0.65)

    # --- LLM and Output ---
    draw_box(ax, 7.0, 0.35, 5.0, 0.55,
             'Groq LLM Explanation  →  Severity + PDF Report',
             facecolor='#FFF7ED', edgecolor='#EA580C', fontsize=9)

    # --- Legend ---
    legend_items = [
        mpatches.Patch(facecolor='#BFDBFE', edgecolor='#2563EB', label='Mexican model (comorbidity)'),
        mpatches.Patch(facecolor='#BBF7D0', edgecolor='#16A34A', label='Einstein model (lab CBC)'),
        mpatches.Patch(facecolor='#F3E8FF', edgecolor='#7C3AED', label='Ensemble'),
        mpatches.Patch(facecolor='#FEF3C7', edgecolor='#D97706', label='Explainability layer'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=8,
              framealpha=0.9, bbox_to_anchor=(0.01, 0.01))

    ax.set_title('Figure 1: COVID-19 CDSS System Architecture',
                 fontsize=13, fontweight='bold', pad=12)

    out = os.path.join(OUT_DIR, 'fig1_architecture.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')

fig1_architecture()

# ---------------------------------------------------------------------------
# Figure 2 — Model Comparison Bar Chart
# ---------------------------------------------------------------------------

def fig2_model_comparison():
    models = ['Mexican\n(Comorbidity)', 'Einstein\n(Lab CBC)', 'Ensemble\n(60/40)']
    overall_acc = [51.6, 85.1, 86.0]
    severe_recall = [31.0, 33.3, None]

    colors_overall = ['#2563EB', '#16A34A', '#7C3AED']
    colors_severe  = ['#93C5FD', '#86EFAC', '#E9D5FF']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    x = np.arange(len(models))
    bar_w = 0.35

    # Overall accuracy bars
    bars1 = ax.bar(x - bar_w/2, overall_acc, bar_w,
                   color=colors_overall, edgecolor='white', linewidth=1.2,
                   label='Overall Accuracy (%)', zorder=3)

    # Severe recall bars (N/A for ensemble)
    for i, (val, col) in enumerate(zip(severe_recall, colors_severe)):
        if val is not None:
            b = ax.bar(x[i] + bar_w/2, val, bar_w,
                       color=col, edgecolor='white', linewidth=1.2,
                       label='Severe Recall (%)' if i == 0 else '_nolegend_',
                       zorder=3)
            ax.text(x[i] + bar_w/2, val + 0.8, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            # Hatched NA bar for Ensemble
            ax.bar(x[i] + bar_w/2, 20, bar_w,
                   color='#E5E7EB', edgecolor='#9CA3AF', linewidth=1.2,
                   hatch='///', label='_nolegend_', zorder=3)
            ax.text(x[i] + bar_w/2, 10, 'N/A\n(3-class\nmapping)',
                    ha='center', va='center', fontsize=7.5, color='#6B7280',
                    fontstyle='italic')

    # Value labels on overall accuracy bars
    for bar, val in zip(bars1, overall_acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Improvement annotation: Mexican → Ensemble
    ax.annotate('', xy=(x[2] - bar_w/2, 86.5), xytext=(x[0] - bar_w/2, 52.2),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.8,
                                connectionstyle='arc3,rad=-0.25'))
    ax.text(1.0, 72, '+34.4 pp', color='#DC2626', fontsize=10,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2',
                      edgecolor='#DC2626', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    overall_patch = mpatches.Patch(color='#6B7280', label='Overall Accuracy (%)')
    severe_patch  = mpatches.Patch(color='#D1D5DB', label='Severe Class Recall (%)')
    ax.legend(handles=[overall_patch, severe_patch], fontsize=9,
              loc='upper left', framealpha=0.9)

    ax.set_title('Figure 2: Model Performance Comparison\n'
                 'Overall Accuracy and Severe Class Recall',
                 fontsize=12, fontweight='bold')

    out = os.path.join(OUT_DIR, 'fig2_model_comparison.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')

fig2_model_comparison()

# ---------------------------------------------------------------------------
# Figure 3 — Confusion Matrix (Mexican model)
# ---------------------------------------------------------------------------

def fig3_confusion_matrix():
    cm = np.array([
        [1206,   84,  422,   15,  273],
        [  14, 1611,    0,  375,    0],
        [ 626,    0,  993,    2,  379],
        [ 112, 1244,  159,  438,   47],
        [ 199,    0,  202,    0,  180],
    ])
    classes = ['Critical', 'Mild', 'Moderate', 'No_COVID', 'Severe']

    # Row-normalise
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('white')

    for ax_idx, (data, title, fmt) in enumerate([
        (cm,      'Raw Counts',       '{:.0f}'),
        (cm_norm, 'Row-Normalised (%)', '{:.1f}%'),
    ]):
        ax = axes[ax_idx]
        im = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Blues,
                       vmin=0, vmax=data.max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = data.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                color = 'white' if data[i, j] > thresh else 'black'
                ax.text(j, i, fmt.format(data[i, j]),
                        ha='center', va='center', fontsize=8,
                        color=color, fontweight='bold')

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha='right', fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')

    fig.suptitle('Figure 3: Mexican Comorbidity Model — Confusion Matrix\n'
                 'Test set: n=8,581  (2,000 per class, Severe=581)',
                 fontsize=12, fontweight='bold', y=1.02)

    out = os.path.join(OUT_DIR, 'fig3_confusion_matrix.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')

fig3_confusion_matrix()

# ---------------------------------------------------------------------------
# Figure 4 — Einstein Feature Importance (SHAP)
# ---------------------------------------------------------------------------

def fig4_feature_importance():
    features = ['Lymphocytes', 'CRP', 'Neutrophils', 'Leukocytes',
                'Hemoglobin', 'Platelets', 'RDW', 'Monocytes',
                'MCV', 'Red Blood Cells', 'Eosinophils']
    shap_vals  = [0.312, 0.287, 0.198, 0.176, 0.145, 0.132, 0.098,
                  0.087, 0.071, 0.054, 0.038]
    directions = [-1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1]
    colors     = ['#2563EB' if d < 0 else '#DC2626' for d in directions]

    # Reverse so highest is at top
    features_r  = list(reversed(features))
    shap_r      = list(reversed(shap_vals))
    colors_r    = list(reversed(colors))

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')

    y_pos = np.arange(len(features_r))
    bars = ax.barh(y_pos, shap_r, color=colors_r, edgecolor='white',
                   linewidth=0.8, height=0.65)

    for bar, val in zip(bars, shap_r):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left', fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_r, fontsize=10)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_xlim(0, max(shap_r) * 1.18)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    red_patch  = mpatches.Patch(color='#DC2626', label='Increases severity risk')
    blue_patch = mpatches.Patch(color='#2563EB', label='Protective (decreases severity)')
    ax.legend(handles=[red_patch, blue_patch], fontsize=9,
              loc='lower right', framealpha=0.9)

    ax.set_title('Figure 4: Einstein Model — Feature Importance (SHAP)\n'
                 'Mean |SHAP value| across 602 Einstein patients',
                 fontsize=12, fontweight='bold')

    out = os.path.join(OUT_DIR, 'fig4_feature_importance.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')

fig4_feature_importance()

# ---------------------------------------------------------------------------
# Figure 5 — Ensemble Decision Flow (Critical case example)
# ---------------------------------------------------------------------------

def fig5_decision_flow():
    mexican_proba  = {'Critical': 72.6, 'Moderate': 12.5, 'Severe': 12.1,
                      'Mild': 1.8, 'No_COVID': 1.0}
    einstein_proba = {'Severe (→Critical)': 72.0, 'Non_Severe': 20.0, 'No_COVID': 8.0}
    ensemble_confidence = 86.0

    top_shap = [
        ('Clinical setting', +0.279, 'hospitalized'),
        ('Pneumonia',        +0.070, 'present'),
        ('Age risk (>60)',   +0.058, 'yes'),
    ]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[2.2, 1.4],
                           hspace=0.45, wspace=0.35)

    # ---- Panel A: Comorbidity Inputs + Mexican proba ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis('off')
    ax_a.set_title('Mexican Comorbidity Model', fontsize=10,
                   fontweight='bold', color='#1D4ED8', pad=6)

    patient_info = (
        "Patient: 75y/o Male\n"
        "──────────────────────\n"
        "Diabetes:        Yes\n"
        "Hypertension:    Yes\n"
        "Obesity:         Yes\n"
        "Cardiovascular:  Yes\n"
        "Pneumonia:       Yes\n"
        "Clinical setting: ICU\n"
        "──────────────────────\n"
        "Comorbidity count: 4\n"
        "Age risk (>60):   Yes\n"
        "High risk flag:   Yes"
    )
    ax_a.text(0.5, 0.72, patient_info, ha='center', va='center',
              fontsize=8, family='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#DBEAFE',
                        edgecolor='#2563EB', linewidth=1.5))

    # Mexican probability bars
    mx_classes = list(mexican_proba.keys())
    mx_vals    = list(mexican_proba.values())
    bar_colors = ['#DC2626' if c == 'Critical' else '#93C5FD' for c in mx_classes]
    y_pos = np.arange(len(mx_classes))
    ax_a.barh([0.08, 0.14, 0.20, 0.26, 0.32],
              [v/100 * 0.85 for v in mx_vals],
              height=0.05, color=bar_colors, left=0.08)
    for yi, (cls, val) in zip([0.08, 0.14, 0.20, 0.26, 0.32],
                               zip(mx_classes, mx_vals)):
        ax_a.text(0.06, yi, cls, ha='right', va='center', fontsize=7.5)
        ax_a.text(0.08 + val/100 * 0.85 + 0.01, yi,
                  f'{val:.1f}%', ha='left', va='center', fontsize=7.5,
                  fontweight='bold' if cls == 'Critical' else 'normal',
                  color='#DC2626' if cls == 'Critical' else 'black')

    ax_a.text(0.5, 0.01, '→ Primary: CRITICAL (72.6%)',
              ha='center', va='bottom', fontsize=9, fontweight='bold',
              color='#DC2626')

    # ---- Panel B: Einstein CBC + Einstein proba ----
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis('off')
    ax_b.set_title('Einstein Lab Model (CBC)', fontsize=10,
                   fontweight='bold', color='#15803D', pad=6)

    cbc_info = (
        "CBC Panel Results\n"
        "──────────────────────\n"
        "Lymphocytes:  ↓ 0.8\n"
        "Neutrophils:  ↑ 11.2\n"
        "CRP:          ↑ 142\n"
        "Hemoglobin:   ↓ 10.1\n"
        "Leukocytes:   ↑ 14.3\n"
        "Platelets:    ↓ 142\n"
        "RDW:          ↑ 16.2\n"
        "──────────────────────\n"
        "(11 features total)"
    )
    ax_b.text(0.5, 0.67, cbc_info, ha='center', va='center',
              fontsize=8, family='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#DCFCE7',
                        edgecolor='#16A34A', linewidth=1.5, linestyle='dashed'))

    # Einstein probability bars
    ein_classes = list(einstein_proba.keys())
    ein_vals    = list(einstein_proba.values())
    bar_colors_e = ['#DC2626' if 'Severe' in c else '#86EFAC' for c in ein_classes]
    for yi, (cls, val) in zip([0.12, 0.19, 0.26], zip(ein_classes, ein_vals)):
        ax_b.barh([yi], [val/100 * 0.85], height=0.05,
                  color='#DC2626' if 'Severe' in cls else '#86EFAC',
                  left=0.08)
        ax_b.text(0.06, yi, cls, ha='right', va='center', fontsize=7)
        ax_b.text(0.08 + val/100 * 0.85 + 0.01, yi,
                  f'{val:.1f}%', ha='left', va='center', fontsize=7.5,
                  fontweight='bold' if 'Severe' in cls else 'normal',
                  color='#DC2626' if 'Severe' in cls else 'black')

    ax_b.text(0.5, 0.04, '→ Primary: SEVERE (72.0%)',
              ha='center', va='bottom', fontsize=9, fontweight='bold',
              color='#DC2626')

    # ---- Panel C: Ensemble Output ----
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis('off')
    ax_c.set_title('Ensemble Decision Output  (60% Mexican + 40% Einstein)',
                   fontsize=11, fontweight='bold', color='#6D28D9', pad=6)

    # Big CRITICAL badge
    ax_c.text(0.12, 0.55, 'CRITICAL', ha='center', va='center',
              fontsize=22, fontweight='bold', color='white',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#DC2626',
                        edgecolor='#991B1B', linewidth=2))

    # Confidence bar
    ax_c.text(0.30, 0.82, f'Ensemble Confidence: {ensemble_confidence:.0f}%',
              ha='left', va='center', fontsize=10, fontweight='bold')
    ax_c.barh([0.65], [ensemble_confidence/100 * 0.45], height=0.12,
              color='#DC2626', left=0.30, alpha=0.85)
    ax_c.barh([0.65], [0.45], height=0.12,
              color='#FEE2E2', left=0.30, zorder=0)
    ax_c.text(0.30 + ensemble_confidence/100 * 0.45 + 0.01, 0.65,
              f'{ensemble_confidence:.0f}%', va='center', fontsize=10,
              fontweight='bold', color='#991B1B')

    # SHAP explanation
    ax_c.text(0.78, 0.85, 'Top SHAP Drivers:', ha='center', va='center',
              fontsize=9, fontweight='bold', color='#374151')
    for i, (feat, val, note) in enumerate(top_shap):
        y = 0.67 - i * 0.14
        color = '#DC2626' if val > 0 else '#2563EB'
        sign  = '+' if val > 0 else ''
        ax_c.text(0.62, y, f'{sign}{val:+.3f}', ha='left', va='center',
                  fontsize=9, fontweight='bold', color=color)
        ax_c.text(0.70, y, f'{feat} ({note})', ha='left', va='center',
                  fontsize=9, color='#374151')

    # Recommendation box
    ax_c.text(0.50, 0.18,
              'Recommendation: Immediate ICU admission — '
              'monitor respiratory function, consider mechanical ventilation protocol.',
              ha='center', va='center', fontsize=8.5, color='#1F2937',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF3C7',
                        edgecolor='#D97706', linewidth=1.5))

    # Arrows from panels to ensemble
    fig.add_artist(mpatches.FancyArrowPatch(
        posA=(0.22, 0.42), posB=(0.35, 0.12),
        arrowstyle='->', color='#6D28D9', lw=2,
        transform=fig.transFigure, mutation_scale=15))
    fig.add_artist(mpatches.FancyArrowPatch(
        posA=(0.78, 0.42), posB=(0.65, 0.12),
        arrowstyle='->', color='#6D28D9', lw=2,
        transform=fig.transFigure, mutation_scale=15))

    fig.suptitle('Figure 5: Ensemble Decision Flow — Critical Case Example\n'
                 'Patient: 75y/o, 4 comorbidities, pneumonia, ICU setting',
                 fontsize=12, fontweight='bold', y=0.98)

    out = os.path.join(OUT_DIR, 'fig5_decision_flow.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')

fig5_decision_flow()

# ---------------------------------------------------------------------------
# OWID figures (static, used in paper)
# ---------------------------------------------------------------------------
try:
    import urllib.request
    owid_total_cases  = "676 million"
    owid_total_deaths = "6.9 million"
    owid_mexico_deaths = "334,000"
    owid_brazil_deaths = "702,000"
except Exception:
    pass

print("\nAll 5 figures generated successfully.")
