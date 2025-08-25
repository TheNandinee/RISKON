import pandas as pd
import numpy as np
import pickle
import os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import imageio # For creating GIFs
from tqdm import tqdm

print("--- RISKON Historical Risk Trajectory Generator ---")

# ===================================================================
# 1. Configuration & Artifact Loading
# ===================================================================
# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
ID_VARIABLE = 'SK_ID_CURR'
TIME_VARIABLE = 'MONTHS_BEFORE_APPLICATION'
ANIMATION_DIR = os.path.join(ARTIFACTS_DIR, 'animation_frames')

# --- Create directory for animation frames ---
os.makedirs(ANIMATION_DIR, exist_ok=True)

# --- Load all saved artifacts ---
print("Loading all trained model artifacts...")
try:
    with open(os.path.join(ARTIFACTS_DIR, 'final_elasticnet_models.pkl'), 'rb') as f:
        final_models = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'ode_params.pkl'), 'rb') as f:
        ode_params = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'kmeans_model.pkl'), 'rb') as f:
        kmeans_model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'woe_binner_objects.pkl'), 'rb') as f:
        binner_objects = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'selected_features.pkl'), 'rb') as f:
        selected_features = pickle.load(f)
    print("Artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading artifacts: {e}. Please ensure the full training pipeline has been run.")
    exit()

# --- Load required datasets ---
print("Loading historical data...")
try:
    df_raw = pd.read_csv(os.path.join(BASE_DIR, 'riskon_combined_dataset.csv'))
    df_cohorted = pd.read_csv(os.path.join(BASE_DIR, 'riskon_cohorted_dataset.csv'))
    
    # Merge the two to have all necessary columns
    df_merged = pd.merge(
        df_cohorted,
        df_raw.drop(columns=['TARGET'], errors='ignore'),
        on=[ID_VARIABLE, TIME_VARIABLE],
        how='left'
    ).sort_values([ID_VARIABLE, TIME_VARIABLE])
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data files: {e}.")
    exit()

# ===================================================================
# 2. Core ODE Solver Functions (Replicated from training)
# ===================================================================

def riskon_ode_system(t, y, params, applicant_data, time_points):
    """Defines the system of differential equations."""
    idx = np.abs(time_points - t).argmin()
    data_t = applicant_data.iloc[idx]
    d_sigma_dt = params['kappa'] * (data_t['RISKON_INCOME_VOLATILITY'] - y[0])
    d_R_dt = -params['lambda'] * y[1] + params['omega'] * data_t['RISKON_RECOVERY_RATE_T']
    return [d_sigma_dt, d_R_dt]

def solve_for_applicant_at_time_t(applicant_data_slice):
    """Solves the ODEs for a given historical slice of data."""
    if len(applicant_data_slice) < 2:
        return pd.Series([
            applicant_data_slice['RISKON_INCOME_VOLATILITY'].iloc[0],
            applicant_data_slice['RISKON_RECOVERY_RATE_T'].iloc[0]
        ], index=['ODE_Income_Volatility', 'ODE_Recovery_Rate'])

    cohort = applicant_data_slice['FINANCIAL_COHORT'].iloc[0]
    params = ode_params[cohort]
    time_points = applicant_data_slice[TIME_VARIABLE].values
    y0 = [applicant_data_slice['RISKON_INCOME_VOLATILITY'].iloc[0], applicant_data_slice['RISKON_RECOVERY_RATE_T'].iloc[0]]
    
    sol = solve_ivp(fun=riskon_ode_system, t_span=(time_points.min(), time_points.max()), y0=y0,
                    method='RK45', t_eval=time_points, args=(params, applicant_data_slice, time_points))
    
    if not sol.success:
        return pd.Series([np.nan, np.nan], index=['ODE_Income_Volatility', 'ODE_Recovery_Rate'])

    return pd.Series([sol.y[0, -1], sol.y[1, -1]], index=['ODE_Income_Volatility', 'ODE_Recovery_Rate'])

# ===================================================================
# 3. Main Trajectory Generation Function
# ===================================================================

def generate_historical_risk_trajectory(applicant_id, full_data):
    """
    Calculates the historical probability of default for a single applicant.
    """
    applicant_history = full_data[full_data[ID_VARIABLE] == applicant_id].copy()
    
    if applicant_history.empty:
        return None

    cohort_id = applicant_history['FINANCIAL_COHORT'].iloc[0]
    model = final_models[cohort_id]
    
    risk_trajectory = []
    
    for i in range(1, len(applicant_history) + 1):
        current_moment_data = applicant_history.iloc[:i]
        latest_snapshot = current_moment_data.iloc[-1]
        ode_features_at_t = solve_for_applicant_at_time_t(current_moment_data)
        
        features_at_t = latest_snapshot.drop(['TARGET', 'SK_ID_CURR', 'FINANCIAL_COHORT'], errors='ignore')
        features_at_t['ODE_Income_Volatility'] = ode_features_at_t['ODE_Income_Volatility']
        features_at_t['ODE_Recovery_Rate'] = ode_features_at_t['ODE_Recovery_Rate']
        
        model_features = model.feature_names_in_
        features_at_t = features_at_t.reindex(model_features).fillna(0) # Ensure all features are present

        raw_score = model.predict(pd.DataFrame([features_at_t]))[0]
        probability = 1 / (1 + np.exp(-raw_score))
        
        risk_trajectory.append({
            'Months Before Application': latest_snapshot[TIME_VARIABLE],
            'Probability of Default': probability
        })
        
    return pd.DataFrame(risk_trajectory)

# ===================================================================
# 4. Example Usage and Visualization
# ===================================================================
if __name__ == "__main__":
    # --- Select 4 most volatile applicants from the same cohort ---
    print("\nFinding most volatile applicants for visualization...")
    
    applicant_counts = df_merged[ID_VARIABLE].value_counts()
    eligible_ids = applicant_counts[applicant_counts > 15].index
    
    volatility_data = []
    for app_id in tqdm(eligible_ids[:1000], desc="Calculating Volatility"): # Sample for speed
        traj = generate_historical_risk_trajectory(app_id, df_merged)
        if traj is not None and not traj.empty:
            volatility = traj['Probability of Default'].std()
            cohort = df_merged[df_merged[ID_VARIABLE] == app_id]['FINANCIAL_COHORT'].iloc[0]
            volatility_data.append({'ID': app_id, 'Volatility': volatility, 'Cohort': cohort})
            
    volatility_df = pd.DataFrame(volatility_data)

    if volatility_df.empty:
        print("Could not find any applicants with sufficient history. Exiting.")
        exit()

    most_volatile_cohort = volatility_df.groupby('Cohort')['Volatility'].mean().idxmax()
    sample_applicants = volatility_df[volatility_df['Cohort'] == most_volatile_cohort].nlargest(4, 'Volatility')
    
    print(f"\nSelected Cohort {most_volatile_cohort} for plotting.")
    print("Top 4 most volatile applicants:")
    print(sample_applicants)

    # --- Generate trajectory data for the selected applicants ---
    trajectories = {}
    for app_id in sample_applicants['ID']:
        trajectories[app_id] = generate_historical_risk_trajectory(app_id, df_merged)

    # --- Create and save the animated GIF with a night theme ---
    print("\nCreating animated GIF...")
    filenames = []
    max_len = max(len(t) for t in trajectories.values()) if trajectories else 0
    colors = plt.cm.spring(np.linspace(0, 1, 4)) # Neon colors

    for i in range(1, max_len + 1):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for idx, (app_id, traj) in enumerate(trajectories.items()):
            ax.plot(
                traj['Months Before Application'][:i],
                traj['Probability of Default'][:i],
                marker='o', markersize=6, linewidth=2.5,
                color=colors[idx], label=f'Applicant {app_id}',
                path_effects=[plt.matplotlib.patheffects.withSimplePatchShadow(offset=(2,-2), shadow_rgb_alpha=(1,1,1,0.2), alpha=1)]
            )

        ax.set_title(f'Historical Probability of Default (Cohort {most_volatile_cohort})', fontsize=20, color='white', pad=20)
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
        plt.savefig(filename, bbox_inches='tight', facecolor='#121212')
        plt.close()

    gif_path = os.path.join(ARTIFACTS_DIR, f'trajectory_cohort_{most_volatile_cohort}.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.2, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        for _ in range(10):
             writer.append_data(image)

    for filename in filenames:
        os.remove(filename)

    print(f"\nAnimated trajectory graph saved to: '{gif_path}'")
