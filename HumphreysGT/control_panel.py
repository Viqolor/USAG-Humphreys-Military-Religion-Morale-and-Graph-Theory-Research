import pandas as pd
import matplotlib.pyplot as plt
from research_engine import GraphProcessor, SimulationConfig, MoraleSimulator

# 1. Load Data
dist_df = pd.read_csv('distances.csv')
fac_df = pd.read_csv('facilities.csv')
blk_df = pd.read_csv('blocks.csv')

# 2. Initialize
gp = GraphProcessor(A=2.0, B=0.1)
cfg = SimulationConfig()
sim = MoraleSimulator(cfg)

# --- BASELINE SCENARIO ---
print("Running Baseline Scenario...")
base_eb, _ = gp.calculate_influence(dist_df, fac_df, blk_df)
baseline_morale = sim.predict_morale(base_eb)

# --- EXPERIMENT 1: GLOBAL REMOVAL ---
print("Running 'No Religion' Scenario...")
no_rf_eb = base_eb.copy()
rel_col = [c for c in no_rf_eb.columns if 'rel' in c.lower()][0]
no_rf_eb[rel_col] = 0
no_rf_morale = sim.predict_morale(no_rf_eb)

# --- EXPERIMENT 2: INDIVIDUAL NODE SWAPS ---
# Identify all religious nodes from your facility list
religious_nodes = fac_df[fac_df['type'] == 'religious']['facility_id'].tolist()
swap_results = {}

for node_id in religious_nodes:
    print(f"Running Swap Experiment for: {node_id}...")
    exp_fac = fac_df.copy()
    exp_fac.loc[exp_fac['facility_id'] == node_id, 'type'] = 'secular'
    
    eb_df, _ = gp.calculate_influence(dist_df, exp_fac, blk_df)
    swap_results[node_id] = sim.predict_morale(eb_df)

# --- EXPERIMENT 3: ADDING NEW RF AT ZG ---
print("Running New Node at ZG Experiments...")
new_node_results = {}
zg_dist_template = dist_df[dist_df['facility_id'] == 'ZG'].copy()

for rf_id in religious_nodes:
    # Fetch attributes from facilities.csv
    rf_template = fac_df[fac_df['facility_id'] == rf_id].iloc[0]
    new_node_id = f"NEW_RF_{rf_id}_at_ZG"
    
    # Create temporary DataFrames for this iteration
    new_fac_row = pd.DataFrame([{
        'facility_id': new_node_id,
        'type': 'religious',
        'personhrs': rf_template['personhrs'],
        'quality': rf_template['quality']
    }])
    exp3_fac = pd.concat([fac_df, new_fac_row], ignore_index=True)

    new_dist_rows = zg_dist_template.copy()
    new_dist_rows['facility_id'] = new_node_id
    exp3_dist = pd.concat([dist_df, new_dist_rows], ignore_index=True)

    # Run Simulation
    eb_df, _ = gp.calculate_influence(exp3_dist, exp3_fac, blk_df)
    new_node_results[rf_id] = sim.predict_morale(eb_df)

# --- OUTPUT: DETAILED SUMMARIES ---

def print_block_summary(title, morale_df):
    print(f"\n" + "="*50)
    print(f"{title.upper()} - NODE-LEVEL MORALE")
    print("="*50)
    # Transpose (.T) makes it easier to read: Blocks as rows, Stats as columns
    summary = morale_df.describe().T[['mean', 'std', '25%', '75%']]
    print(summary)
    print(f"Global Mean: {morale_df.mean().mean():.4f}")
    print("="*50)

# --- FINAL ORDERED REPORTING WITH SIGNIFICANCE ---

def print_sig_report(name, stats_dict):
    # Pull the global metrics for terminal display
    # This prevents the KeyError by accessing the correct sub-dictionary
    global_stats = stats_dict['global']
    
    print(f"\n--- Significance Test: {name} ---")
    print(f"Global Mean Delta: {global_stats['mean_delta']:+.4f}")
    print(f"Global P-Value:    {global_stats['p_value']:.4e}")
    print(f"Global Cohen's d:  {global_stats['cohens_d']:.4f}")
    print(f"Global 95% CI:     [{global_stats['ci_lower']:.4f}, {global_stats['ci_upper']:.4f}]")
    
    # Interpretation based on the Global Cohen's d
    effect = "Negligible"
    d_abs = abs(global_stats['cohens_d'])
    if d_abs > 0.8: effect = "Large"
    elif d_abs > 0.5: effect = "Medium"
    elif d_abs > 0.2: effect = "Small"
    print(f"Global Effect:     {effect}")

# 1. Baseline Summary
print_block_summary("EXPERIMENT 1: BASELINE", baseline_morale)

# 2. Experiment 2: Global Removal
print_block_summary("EXPERIMENT 2: NO RELIGIOUS FACILITIES", no_rf_morale)
sig_2 = sim.run_significance_test(baseline_morale, no_rf_morale)
print_sig_report("Removal Impact", sig_2)

# 3. Experiment 3: Individual Swaps
print("\n" + "="*50)
print("EXPERIMENT 3: FACILITY SWAPS (RELIGIOUS NODE -> SECULAR NODE)")
print("="*50)
for node_id, morale_df in swap_results.items():
    print_block_summary(f"Swap {node_id}", morale_df)
    sig_3 = sim.run_significance_test(baseline_morale, morale_df)
    print_sig_report(f"Swap {node_id} Significance", sig_3)

# 4. Experiment 4: New Node at ZG
print("\n" + "="*50)
print("EXPERIMENT 4: NEW RELIGIOUS NODE AT ZG")
print("="*50)
for rf_id, morale_df in new_node_results.items():
    print_block_summary(f"New Node ({rf_id} specs)", morale_df)
    sig_4 = sim.run_significance_test(baseline_morale, morale_df)
    print_sig_report(f"Addition ({rf_id} specs) Significance", sig_4)

# --- MASTER EXPORT TO CSV ---
print("\nExporting all results to master_results.csv...")

# --- MASTER EXPORT TO CSV (with Node-Level Significance) ---

export_data = []

def create_record(name, morale_df, sig_results=None):
    record = {
        'Scenario': name,
        'Global_Mean': morale_df.mean().mean()
    }
    
    # Add mean morale for each block
    for block_id in morale_df.columns:
        record[f'{block_id}_Mean'] = morale_df[block_id].mean()
    
    if sig_results:
        # Global stats
        record['Global_P_Value'] = sig_results['global']['p_value']
        record['Global_Cohens_d'] = sig_results['global']['cohens_d']
        record['Global_CI_Lower'] = sig_results['global']['ci_lower']
        record['Global_CI_Upper'] = sig_results['global']['ci_upper']
        
        # Individual Node stats
        for block_id, stats in sig_results['nodes'].items():
            record[f'{block_id}_P_Value'] = stats['p_value']
            record[f'{block_id}_Cohens_d'] = stats['cohens_d']
            record[f'{block_id}_CI_Lower'] = stats['ci_lower']
            record[f'{block_id}_CI_Upper'] = stats['ci_upper']
            
    return record

# --- Collection Logic ---

# Baseline
export_data.append(create_record("Baseline", baseline_morale))

# Exp 2
sig2 = sim.run_significance_test(baseline_morale, no_rf_morale)
export_data.append(create_record("Exp 2: Removal", no_rf_morale, sig2))

# Exp 3
for node_id, m_df in swap_results.items():
    sig3 = sim.run_significance_test(baseline_morale, m_df)
    export_data.append(create_record(f"Exp 3: Swap {node_id}", m_df, sig3))

# Exp 4
for rf_id, m_df in new_node_results.items():
    sig4 = sim.run_significance_test(baseline_morale, m_df)
    export_data.append(create_record(f"Exp 4: New Node ({rf_id} specs)", m_df, sig4))

# Final Save
pd.DataFrame(export_data).to_csv('master_results.csv', index=False)
print("\nSuccess: master_results.csv updated with node-level significance metrics.")

# --- INDIVIDUAL EXPORTS FOR RAWGRAPHS ---
print("\nExporting individual melted CSVs for each scenario...")

def export_individual_scenario(name, morale_df):
    # Melt the data so Block_ID is on the X-axis
    melted = morale_df.melt(
        var_name='Block_ID', 
        value_name='Morale_Value'
    )
    # This keeps exactly 20% of the 10,000 iterations for each unit
    melted = melted.groupby('Block_ID').sample(n=2000, random_state=42)
    # Sanitize filename (remove spaces/special chars)
    filename = f"raw_dist_{name.replace(' ', '_').replace(':', '')}.csv"
    melted.to_csv(filename, index=False)
    print(f"Saved: {filename}")

# 1. Export Baseline
export_individual_scenario("Baseline", baseline_morale)

# 2. Export Exp 2
export_individual_scenario("Exp 2 Removal", no_rf_morale)

# 3. Export Exp 3 Swaps
for node_id, morale_df in swap_results.items():
    export_individual_scenario(f"Exp 3 Swap {node_id}", morale_df)

# 4. Export Exp 4 New Nodes
for rf_id, morale_df in new_node_results.items():
    export_individual_scenario(f"Exp 4 New RF {rf_id}", morale_df)