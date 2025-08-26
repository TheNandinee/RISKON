import pandas as pd
import numpy as np
import pickle
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print(" RISKON  Visualization ")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts_lgbm')
DATA_FILE = os.path.join(BASE_DIR, 'riskon_combined_dataset.csv')
ID_VARIABLE = 'SK_ID_CURR'
TARGET_VARIABLE = 'TARGET'
TIME_VARIABLE = 'MONTHS_BEFORE_APPLICATION'
ANIMATION_DIR = os.path.join(ARTIFACTS_DIR, 'animation_frames')

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(ANIMATION_DIR, exist_ok=True)

print("Loading historical data...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"[+] Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"[!] Data file not found: {DATA_FILE}. Please ensure it exists.")
    exit()

df_snapshot = df.loc[df.groupby(ID_VARIABLE)[TIME_VARIABLE].idxmax()].copy()

print("\nTraining LightGBM Benchmark Model ")

y = df_snapshot[TARGET_VARIABLE]
X = df_snapshot.drop(columns=[TARGET_VARIABLE, ID_VARIABLE])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_lgbm = pd.Series(np.zeros(len(X)), index=X.index)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\n--- Processing Fold {fold+1}/5 ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(random_state=42, n_jobs=4, is_unbalance=True)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(10, verbose=False)])
    
    preds = model.predict_proba(X_val)[:, 1]
    oof_preds_lgbm.iloc[val_idx] = preds

final_auc_lgbm = roc_auc_score(y, oof_preds_lgbm)
print(f"\nFinal LightGBM Model Evaluation ")
print(f"Overall Out-of-Fold AUC ROC Score: {final_auc_lgbm:.4f}")
print("\n Training final LightGBM model on all data for deployment ")
final_lgbm_model = lgb.LGBMClassifier(random_state=42, n_jobs=4, is_unbalance=True)
final_lgbm_model.fit(X, y)

with open(os.path.join(ARTIFACTS_DIR, 'final_lgbm_model.pkl'), 'wb') as f:
    pickle.dump(final_lgbm_model, f)
print(" Final LightGBM model saved to artifacts.")

def generate_lgbm_trajectory(applicant_id, full_data, model):
    """Generate historical risk trajectory using the trained LGBM model."""
    applicant_history = full_data[full_data[ID_VARIABLE] == applicant_id].copy()
    applicant_history = applicant_history.sort_values(TIME_VARIABLE, ascending=False)
    
    if applicant_history.empty:
        return None

    features_for_pred = [col for col in X.columns if col in applicant_history.columns]
    
    predictions = model.predict_proba(applicant_history[features_for_pred])[:, 1]
    
    trajectory_df = pd.DataFrame({
        'Months Before Application': applicant_history[TIME_VARIABLE],
        'Probability of Default': predictions
    })
    
    return trajectory_df.sort_values('Months Before Application', ascending=False)

print("\n--- Generating Trajectories for Visualization ---")
applicant_counts = df[ID_VARIABLE].value_counts()
eligible_ids = applicant_counts[applicant_counts > 15].index
volatility_data = []
sample_ids = np.random.choice(eligible_ids, min(1000, len(eligible_ids)), replace=False)

for app_id in tqdm(sample_ids, desc="Analyzing Volatility"):
    traj = generate_lgbm_trajectory(app_id, df, final_lgbm_model)
    if traj is not None and not traj.empty:
        volatility = traj['Probability of Default'].std()
        volatility_data.append({'ID': app_id, 'Volatility': volatility})

if not volatility_data:
    print("[!] No applicants with sufficient history found for volatility analysis. Exiting.")
    exit()

volatility_df = pd.DataFrame(volatility_data)
sample_applicants = volatility_df.nlargest(4, 'Volatility')

print("\nTop 4 most volatile applicants selected for plotting:")
print(sample_applicants)

trajectories = {row['ID']: generate_lgbm_trajectory(row['ID'], df, final_lgbm_model) for _, row in sample_applicants.iterrows()}

print("\nCreating animated GIF")
filenames = []
max_len = max(len(t) for t in trajectories.values())
colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFFF00'] 

for i in range(1, max_len + 1):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    for idx, (app_id, traj) in enumerate(trajectories.items()):
        if i <= len(traj):
            current_data = traj.iloc[:i]
            color = colors[idx % len(colors)]
            
            ax.plot(current_data['Months Before Application'], current_data['Probability of Default'], color=color, linewidth=10, alpha=0.1, zorder=1)
            ax.plot(current_data['Months Before Application'], current_data['Probability of Default'], color=color, linewidth=6, alpha=0.2, zorder=2)
            ax.plot(current_data['Months Before Application'], current_data['Probability of Default'], color=color, linewidth=2, alpha=1.0, label=f'Applicant {app_id}', zorder=3)
            
            ax.scatter(current_data['Months Before Application'].iloc[-1], current_data['Probability of Default'].iloc[-1], color='white', s=150, alpha=0.5, zorder=4)
            ax.scatter(current_data['Months Before Application'].iloc[-1], current_data['Probability of Default'].iloc[-1], color=color, s=50, zorder=5)


    ax.set_title(f'RISKON - Live Risk Trajectory Benchmark (LightGBM)', fontsize=20, color='white', pad=20)
    ax.set_xlabel('Months Before Application (History)', fontsize=14, color='lightgray')
    ax.set_ylabel('Predicted Probability of Default (%)', fontsize=14, color='lightgray')
    
    max_prob = max(t['Probability of Default'].max() for t in trajectories.values())
    ax.set_ylim(0, max(0.15, max_prob * 1.2))
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.invert_xaxis()
    
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    ax.tick_params(axis='x', colors='lightgray')
    ax.tick_params(axis='y', colors='lightgray')
    legend = ax.legend(facecolor='#222222', framealpha=0.7, fontsize=12, loc='upper left')
    plt.setp(legend.get_texts(), color='white')

    filename = os.path.join(ANIMATION_DIR, f"frame_{i:03d}.png")
    filenames.append(filename)
    plt.savefig(filename, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()

gif_path = os.path.join(ARTIFACTS_DIR, f'lgbm_trajectory_benchmark.gif')
with imageio.get_writer(gif_path, mode='I', duration=0.2, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    for _ in range(10):
        writer.append_data(image)

for filename in filenames:
    os.remove(filename)

print(f"\nAnimated trajectory graph saved to: '{gif_path}'")
print("\nBenchmark and Visualization Complete")