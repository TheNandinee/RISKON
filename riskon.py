import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from optbinning import OptimalBinning
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import roc_auc_score
from scipy.integrate import solve_ivp
from tqdm import tqdm
import json
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(" Starting ")
DATA_FILE = os.path.join(BASE_DIR, 'riskon_combined_dataset.csv')
TARGET_VARIABLE = 'TARGET'
ID_VARIABLE = 'SK_ID_CURR'
TIME_VARIABLE = 'MONTHS_BEFORE_APPLICATION'
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
N_SPLITS = 5
RANDOM_STATE = 42
MIN_IV = 0.02
MAX_IV = 0.5

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded '{DATA_FILE}' with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found. Please run the data pipeline script first.")
    exit()

df_snapshot = df.loc[df.groupby(ID_VARIABLE)[TIME_VARIABLE].idxmax()]
print(f"Using latest snapshot per applicant for WoE training. Total unique applicants: {len(df_snapshot)}")
features_to_transform = [col for col in df.columns if col not in [TARGET_VARIABLE, ID_VARIABLE, TIME_VARIABLE]]

print("\nStarting WoE Transformation and IV Selection ")
binner_objects = {}
selected_features = []

for feature in tqdm(features_to_transform, desc="Processing Features"):
    optb = OptimalBinning(name=feature, solver="cp")
    X_feat = df_snapshot[feature]
    y_feat = df_snapshot[TARGET_VARIABLE]
    optb.fit(X_feat, y_feat)
    binning_table_df = optb.binning_table.build()
    iv_score = binning_table_df['IV'].sum()
    if MIN_IV <= iv_score <= MAX_IV:
        selected_features.append(feature)
        binner_objects[feature] = optb

print(f"\nFeature Selection Complete. Selected {len(selected_features)} features.")
with open(os.path.join(ARTIFACTS_DIR, 'woe_binner_objects.pkl'), 'wb') as f:
    pickle.dump(binner_objects, f)
with open(os.path.join(ARTIFACTS_DIR, 'selected_features.pkl'), 'wb') as f:
    pickle.dump(selected_features, f)
print(f"Saved WoE and feature selection artifacts to '{ARTIFACTS_DIR}'")
df_transformed = df[[ID_VARIABLE, TIME_VARIABLE, TARGET_VARIABLE]].copy()
for feature in selected_features:
    binner = binner_objects[feature]
    df_transformed[f'woe_{feature}'] = binner.transform(df[feature], metric="woe")
transformed_data_path = os.path.join(BASE_DIR, 'riskon_woe_transformed_dataset.csv')
df_transformed.to_csv(transformed_data_path, index=False)
print(f"\nFull WoE transformed dataset saved to: '{transformed_data_path}'")
print("\n Starting RISKON  Unsupervised Cohort Discovery")
df_snapshot_clust = df_transformed.loc[df_transformed.groupby(ID_VARIABLE)[TIME_VARIABLE].idxmax()]
features_for_clustering = [col for col in df_transformed.columns if col.startswith('woe_')]
X_cluster = df_snapshot_clust[features_for_clustering]

print("\nFinding optimal number of cohorts (k) ")
k_range = range(2, 11)
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_cluster)
    score = silhouette_score(X_cluster, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"  - k={k}, Silhouette Score: {score:.4f}")

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of cohorts found: k = {optimal_k}")

final_kmeans_model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
final_kmeans_model.fit(X_cluster)
with open(os.path.join(ARTIFACTS_DIR, 'kmeans_model.pkl'), 'wb') as f:
    pickle.dump(final_kmeans_model, f)
print(f"Final K-Means model artifact saved.")

X_full_cluster = df_transformed[features_for_clustering]
df_transformed['FINANCIAL_COHORT'] = final_kmeans_model.predict(X_full_cluster)
cohorted_data_path = os.path.join(BASE_DIR, 'riskon_cohorted_dataset.csv')
df_transformed.to_csv(cohorted_data_path, index=False)
print(f"Full dataset with cohort assignments saved to: '{cohorted_data_path}'")


print("\nStarting  ODE Calibration & Feature Generation ")

df_cohorted = pd.read_csv(cohorted_data_path)
df_raw = pd.read_csv(DATA_FILE)
df_merged = pd.merge(
    df_cohorted,
    df_raw[[ID_VARIABLE, TIME_VARIABLE, 'RISKON_INCOME_VOLATILITY', 'RISKON_RECOVERY_RATE_T']],
    on=[ID_VARIABLE, TIME_VARIABLE], how='left'
).sort_values([ID_VARIABLE, TIME_VARIABLE])

print("\n Calibrating ODE parameters for each cohort ")
cohort_ids = sorted(df_merged['FINANCIAL_COHORT'].unique())
ode_params = {}

for cohort in cohort_ids:
    print(f"Calibrating for Cohort {cohort}...")
    cohort_data = df_merged[df_merged['FINANCIAL_COHORT'] == cohort].copy()
    
    cohort_data['sigma_prev'] = cohort_data.groupby(ID_VARIABLE)['RISKON_INCOME_VOLATILITY'].shift(1)
    cohort_data['sigma_change'] = cohort_data['RISKON_INCOME_VOLATILITY'] - cohort_data['sigma_prev']
    train_data_sigma = cohort_data.dropna(subset=['sigma_change', 'sigma_prev'])
    
    if not train_data_sigma.empty:
        X_sigma = (train_data_sigma['RISKON_INCOME_VOLATILITY'] - train_data_sigma['sigma_prev']).values.reshape(-1, 1)
        y_sigma = train_data_sigma['sigma_change'].values
        lr_sigma = LinearRegression(fit_intercept=False).fit(X_sigma, y_sigma)
        kappa = lr_sigma.coef_[0] if lr_sigma.coef_[0] > 0 else 0.01
    else:
        kappa = 0.1

    cohort_data['R_prev'] = cohort_data.groupby(ID_VARIABLE)['RISKON_RECOVERY_RATE_T'].shift(1)
    cohort_data['R_change'] = cohort_data['RISKON_RECOVERY_RATE_T'] - cohort_data['R_prev']
    train_data_r = cohort_data.dropna(subset=['R_change', 'R_prev'])
    
    if len(train_data_r) > 2:
        X_r = train_data_r[['R_prev', 'RISKON_RECOVERY_RATE_T']].values
        y_r = train_data_r['R_change'].values
        lr_r = LinearRegression(fit_intercept=False).fit(X_r, y_r)
        lambda_param, omega = -lr_r.coef_[0], lr_r.coef_[1]
    else:
        lambda_param, omega = 0.2, 0.5

    ode_params[cohort] = {
        'kappa': kappa,
        'lambda': max(0.01, lambda_param),
        'omega': max(0, omega)
    }
    print(f"  - Params for Cohort {cohort}: {ode_params[cohort]}")

with open(os.path.join(ARTIFACTS_DIR, 'ode_params.pkl'), 'wb') as f:
    pickle.dump(ode_params, f)
print(f"\nSaved ODE parameters to artifacts.")


def riskon_ode_system(t, y, params, applicant_data, time_points):
    idx = np.abs(time_points - t).argmin()
    data_t = applicant_data.iloc[idx]
    d_sigma_dt = params['kappa'] * (data_t['RISKON_INCOME_VOLATILITY'] - y[0])
    d_R_dt = -params['lambda'] * y[1] + params['omega'] * data_t['RISKON_RECOVERY_RATE_T']
    return [d_sigma_dt, d_R_dt]

def solve_for_applicant(applicant_data):
    if len(applicant_data) < 2:
        return pd.Series([np.nan, np.nan], index=['ODE_Income_Volatility', 'ODE_Recovery_Rate'])

    cohort = applicant_data['FINANCIAL_COHORT'].iloc[0]
    params = ode_params[cohort]
    time_points = applicant_data[TIME_VARIABLE].values
    y0 = [applicant_data['RISKON_INCOME_VOLATILITY'].iloc[0], applicant_data['RISKON_RECOVERY_RATE_T'].iloc[0]]
    
    sol = solve_ivp(fun=riskon_ode_system, t_span=(time_points.min(), time_points.max()), y0=y0,
                    method='RK45', t_eval=time_points, args=(params, applicant_data, time_points))
    
    if not sol.success:
        return pd.Series([np.nan, np.nan], index=['ODE_Income_Volatility', 'ODE_Recovery_Rate'])

    return pd.Series([sol.y[0, -1], sol.y[1, -1]], index=['ODE_Income_Volatility', 'ODE_Recovery_Rate'])

tqdm.pandas(desc="Solving ODEs for each applicant")
ode_features = df_merged.groupby(ID_VARIABLE).progress_apply(solve_for_applicant)
df_snapshot_ode = df_cohorted.loc[df_cohorted.groupby(ID_VARIABLE)[TIME_VARIABLE].idxmax()].set_index(ID_VARIABLE)
df_final_features = df_snapshot_ode.join(ode_features)
final_features_path = os.path.join(BASE_DIR, 'riskon_final_features_dataset.csv')
df_final_features.to_csv(final_features_path)
print(f"\nFinal dataset with ODE features saved to: '{final_features_path}'")
print("\n Starting  Final Model Training & Evaluation")
df_final = pd.read_csv(final_features_path)
df_model = df_final.dropna().copy()
y = df_model[TARGET_VARIABLE]
X = df_model.drop(columns=[TARGET_VARIABLE])
print(f"\n Starting {N_SPLITS}-Fold Cross-Validation for Cohort-Based Modeling ")
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = pd.Series(np.zeros(len(X)), index=X.index)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\n Processing Fold {fold+1}/{N_SPLITS} ")
    X_train, X_val, y_train = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx]
    cohort_models = {}
    for cohort in sorted(X_train['FINANCIAL_COHORT'].unique()):
        train_mask = X_train['FINANCIAL_COHORT'] == cohort
        model = ElasticNetCV(cv=5, random_state=RANDOM_STATE, n_jobs=4)
        model.fit(X_train.loc[train_mask].drop(columns=[ID_VARIABLE, 'FINANCIAL_COHORT']), y_train.loc[train_mask])
        cohort_models[cohort] = model
    
    for cohort, model in cohort_models.items():
        val_mask = X_val['FINANCIAL_COHORT'] == cohort
        if val_mask.any():
            X_val_cohort = X_val.loc[val_mask]
            preds = model.predict(X_val_cohort.drop(columns=[ID_VARIABLE, 'FINANCIAL_COHORT']))
            oof_preds.loc[X_val_cohort.index] = np.clip(preds, 0, 1)

final_auc = roc_auc_score(y, oof_preds)
print(f"\n Final Model Evaluation ")
print(f"Overall Out-of-Fold AUC ROC Score: {final_auc:.4f}")
print("\n Training final models on the entire dataset for deployment")
final_models = {}
for cohort in sorted(X['FINANCIAL_COHORT'].unique()):
    full_mask = X['FINANCIAL_COHORT'] == cohort
    model = ElasticNetCV(cv=5, random_state=RANDOM_STATE, n_jobs=4)
    model.fit(X.loc[full_mask].drop(columns=[ID_VARIABLE, 'FINANCIAL_COHORT']), y.loc[full_mask])
    final_models[cohort] = model

with open(os.path.join(ARTIFACTS_DIR, 'final_elasticnet_models.pkl'), 'wb') as f:
    pickle.dump(final_models, f)
print(f"\nFinal dictionary of trained ElasticNet models saved.")

print("\nStarting Monitoring ")
MONITORING_DIR = os.path.join(ARTIFACTS_DIR, 'monitoring')
os.makedirs(MONITORING_DIR, exist_ok=True)
BASELINE_STATS_PATH = os.path.join(MONITORING_DIR, 'baseline_feature_distributions.pkl')
PSI_REPORT_PATH = os.path.join(MONITORING_DIR, 'psi_report.csv')
RETRAIN_TRIGGER_PATH = os.path.join(MONITORING_DIR, 'retrain_trigger.json')

psi_features = [c for c in df_model.columns if c.startswith('woe_')]

def compute_distribution(series, num_bins=10):
    counts, bins = np.histogram(series.dropna(), bins=num_bins)
    percents = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    return {'bins': bins.tolist(), 'percents': percents.tolist()}

def compute_psi(expected, actual, epsilon=1e-6):
    return float(np.sum((actual - expected) * np.log((actual + epsilon) / (expected + epsilon))))

baseline_distributions = {feat: compute_distribution(df_model[feat]) for feat in psi_features}
with open(BASELINE_STATS_PATH, 'wb') as f:
    pickle.dump(baseline_distributions, f)
print(f"Saved baseline feature distributions for PSI.")
live_window = df_model.tail(max(1000, int(0.1 * len(df_model))))
psi_rows = []
for feat in psi_features:
    base_bins = np.array(baseline_distributions[feat]['bins'])
    counts, _ = np.histogram(live_window[feat].dropna(), bins=base_bins)
    percents = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    psi_val = compute_psi(np.array(baseline_distributions[feat]['percents']), percents)
    psi_rows.append({'feature': feat, 'psi': psi_val})

psi_df = pd.DataFrame(psi_rows).sort_values('psi', ascending=False)
psi_df.to_csv(PSI_REPORT_PATH, index=False)
print(f"PSI report written.")
max_psi = psi_df['psi'].max()
timestamp = datetime.now(timezone.utc).isoformat()
trigger_payload = {
    'timestamp': timestamp,
    'max_psi': max_psi,
    'live_auc': final_auc,
    'retrain': bool(max_psi > 0.25 or final_auc < 0.75),
    'reasons': [r for r in [f"max_psi_exceeds_0.25" if max_psi > 0.25 else "", f"live_auc_below_0.75" if final_auc < 0.75 else ""] if r]
}
with open(RETRAIN_TRIGGER_PATH, 'w') as f:
    json.dump(trigger_payload, f, indent=2)
print("Retraining trigger written:", trigger_payload)
print("\nRISKON Framework Implementation Complete")

