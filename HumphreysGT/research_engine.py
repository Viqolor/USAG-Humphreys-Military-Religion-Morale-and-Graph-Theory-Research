import pandas as pd
import numpy as np
from scipy import stats

class GraphProcessor:
    def __init__(self, A=2.0, B=0.1):
        self.A = A
        self.B = B

    def calculate_influence(self, distances, facilities, blocks):
        # This prevents experiment changes from affecting the baseline
        fac = facilities.copy()
        dist = distances.copy()
        blk = blocks.copy()
        
        fac['type'] = fac['type'].astype(str).str.strip().str.lower()

        df = pd.merge(distances, facilities, on='facility_id', how='left')
        df = pd.merge(df, blocks, on='block_id', how='left')
        df = df.dropna(subset=['personhrs', 'quality', 'barracks_count'])

        # Calculate W_ur (Edge Weights)
        df['W_ur'] = (
            ((1 + self.B * df['quality']) * np.exp(-self.A * df['dist_km']) * df['personhrs']) / df['barracks_count']
        )

        # We group by block AND type to get the sum of weights for each category
        grouped = df.groupby(['block_id', 'type'])['W_ur'].sum().reset_index()

        inf_scores = grouped.pivot(index='block_id', columns='type', values='W_ur').fillna(0)
        
        rename_dict = {}
        for col in inf_scores.columns:
            if 'rel' in col:
                rename_dict[col] = 'E_b_religion'
            elif 'sec' in col:
                rename_dict[col] = 'E_b_secular'
        
        inf_scores = inf_scores.rename(columns=rename_dict)
        inf_scores = inf_scores.reset_index()

        # Final check: Ensure E_b_religion and E_b_secular exist
        for col in ['E_b_religion', 'E_b_secular']:
            if col not in inf_scores.columns:
                inf_scores[col] = 0.0
        
        return inf_scores, df

class SimulationConfig:
    def __init__(self):
        self.b0_mu, self.b0_sigma = 4.0, 0.5
        self.b1_mu = (2.0 / 160.0)
        self.b1_sigma = self.b1_mu / 5
        self.b2_mu = (4.0 / 1766.0)
        self.b2_sigma = self.b2_mu / 5
        self.noise_sigma = 0.5
        self.n_iterations = 10000

class MoraleSimulator:
    def __init__(self, config):
        self.config = config

    def predict_morale(self, eb_df):
        # 1. Ensure there are no NaNs in the input just in case
        eb_df = eb_df.fillna(0)
        
        # 2. Dynamically find columns
        rel_col = [c for c in eb_df.columns if 'rel' in c.lower()][0]
        sec_col = [c for c in eb_df.columns if 'sec' in c.lower() or 'rec' in c.lower()][0]

        # 3. Convert Influence scores to raw NumPy arrays for the math
        # This prevents index misalignment errors
        rel_values = eb_df[rel_col].values
        sec_values = eb_df[sec_col].values

        n = self.config.n_iterations
        b0 = np.random.normal(self.config.b0_mu, self.config.b0_sigma, n)
        b1 = np.random.normal(self.config.b1_mu, self.config.b1_sigma, n)
        b2 = np.random.normal(self.config.b2_mu, self.config.b2_sigma, n)
        
        results = []
        for i in range(n):
            # 4. Perform math using arrays instead of Series
            noise = np.random.normal(0, self.config.noise_sigma, size=len(eb_df))
            y = b0[i] + (b1[i] * rel_values) + (b2[i] * sec_values) + noise
            
            results.append(np.clip(y, 0, 10))
        
        # 5. Build final DataFrame with explicit column names
        return pd.DataFrame(results, columns=eb_df['block_id'].values)

    def run_significance_test(self, baseline_df, experiment_df):
        results = {'global': {}, 'nodes': {}}
        n_obs = len(baseline_df) # Usually 10,000

        def get_stats(base, exp):
            mu_diff = np.mean(exp) - np.mean(base)
            t_stat, p_val = stats.ttest_ind(base, exp)
            
            # Cohen's d
            sd_pooled = np.sqrt((np.var(base) + np.var(exp)) / 2)
            d = mu_diff / sd_pooled if sd_pooled != 0 else 0
            
            # 95% Confidence Interval for the Difference
            # Using Welch's t-interval approach for the difference of means
            std_err = np.sqrt(np.var(base)/n_obs + np.var(exp)/n_obs)
            ci_low, ci_high = stats.t.interval(0.95, df=2*n_obs-2, loc=mu_diff, scale=std_err)
            
            return {
                'p_value': p_val,
                'cohens_d': d,
                'mean_delta': mu_diff,
                'ci_lower': ci_low,
                'ci_upper': ci_high
            }

        # --- 1. Global Significance ---
        results['global'] = get_stats(baseline_df.mean(axis=1), experiment_df.mean(axis=1))

        # --- 2. Node-Level Significance ---
        for block_id in baseline_df.columns:
            results['nodes'][block_id] = get_stats(baseline_df[block_id], experiment_df[block_id])

        return results