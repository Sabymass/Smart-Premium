# src/notebooks/SmartPremium_EDA.py
"""
SmartPremium EDA helper module.

Provides run_eda(train_path, out_dir, target_plots) which:
 - Loads the train CSV
 - Generates 13 plots (saved into a single multi-page PDF)
 - Saves missing-values CSV, correlation CSV, and a small summary text
 - Returns list of artifact paths

Can be run as script (python -m src.notebooks.SmartPremium_EDA)
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import json

DEFAULT_NUM_PLOTS = [
    'Age','Annual Income','Health Score','Previous Claims',
    'Vehicle Age','Credit Score','Insurance Duration','Premium Amount'
]
DEFAULT_CAT_PLOTS = ['Gender', 'Location', 'Policy Type']

def run_eda(train_path='src/data/train.csv', out_dir='outputs/eda', target_plots: int = 13):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    df = pd.read_csv(train_path)
    artifacts = []

    # Save simple summary
    summary = {
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'columns': list(df.columns)
    }
    summary_path = out_dir / 'eda_summary.txt'
    with open(summary_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(summary, indent=2))
    artifacts.append(str(summary_path))

    # Missing values CSV
    missing = df.isna().sum().sort_values(ascending=False)
    missing_path = out_dir / 'missing_values.csv'
    missing.to_csv(missing_path, header=['missing_count'])
    artifacts.append(str(missing_path))

    # Correlation matrix (numeric)
    num = df.select_dtypes(include='number')
    corr_path = out_dir / 'corr_matrix.csv'
    if 'Premium Amount' in num.columns:
        corr = num.corr()
        corr.to_csv(corr_path)
        artifacts.append(str(corr_path))
    else:
        # still save empty
        num.corr().to_csv(corr_path)
        artifacts.append(str(corr_path))

    pdf_path = out_dir / 'eda_plots.pdf'
    count = 0

    # Build ordered list of plots to create
    plots_to_make = []
    # numeric histograms (preferred order)
    for col in DEFAULT_NUM_PLOTS:
        if col in df.columns:
            plots_to_make.append(('hist', col))
    # categorical counts
    for col in DEFAULT_CAT_PLOTS:
        if col in df.columns:
            plots_to_make.append(('bar', col))
    # correlation heatmap
    plots_to_make.append(('corr', None))
    # missing values bar
    plots_to_make.append(('missing', None))

    # If list < target_plots, add boxplots for most important numeric cols until reach target
    extras = [c for c in df.select_dtypes(include='number').columns if c not in DEFAULT_NUM_PLOTS]
    for e in extras:
        if len(plots_to_make) >= target_plots:
            break
        plots_to_make.append(('box', e))

    # If still not enough, add placeholders (rare)
    while len(plots_to_make) < target_plots:
        plots_to_make.append(('placeholder', f'placeholder_{len(plots_to_make)+1}'))

    total = min(len(plots_to_make), target_plots)

    print(f"Generating {total} plots into {pdf_path} ...")
    with PdfPages(pdf_path) as pdf:
        for i, (ptype, col) in enumerate(plots_to_make[:target_plots], start=1):
            fig, ax = plt.subplots(figsize=(8, 4))
            if ptype == 'hist':
                sns.histplot(df[col].dropna(), bins=50, kde=True, ax=ax)
                ax.set_title(f'{col} distribution')
            elif ptype == 'bar':
                df[col].fillna('Missing').value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'{col} counts')
                ax.set_xlabel('')
            elif ptype == 'corr':
                if num.shape[1] > 1:
                    sns.heatmap(num.corr(), annot=True, fmt='.2f', ax=ax, cbar=True)
                    ax.set_title('Numeric correlation matrix')
                else:
                    ax.text(0.5, 0.5, 'Not enough numeric columns for correlation', ha='center', va='center')
                    ax.set_axis_off()
            elif ptype == 'missing':
                missing_top = missing.head(20)
                missing_top.plot(kind='bar', ax=ax)
                ax.set_title('Missing values (top 20 columns)')
                ax.set_xlabel('')
            elif ptype == 'box':
                sns.boxplot(x=df[col].dropna(), ax=ax)
                ax.set_title(f'{col} boxplot')
            else:  # placeholder
                ax.text(0.5, 0.5, f'Placeholder plot {i}', ha='center', va='center')
                ax.set_axis_off()

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            count += 1
            print(f"  Saved {count}/{total}: {ptype} {col if col else ''}")

    artifacts.append(str(pdf_path))
    print(f"Saved PDF with {count} pages -> {pdf_path}")

    # Return artifact list
    return artifacts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='src/data/train.csv')
    parser.add_argument('--out', default='outputs/eda')
    parser.add_argument('--plots', type=int, default=13)
    args = parser.parse_args()
    run_eda(train_path=args.train, out_dir=args.out, target_plots=args.plots)
