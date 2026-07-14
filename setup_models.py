"""
setup_models.py — Run this ONCE to train and save models on your machine.

Usage:
    python setup_models.py --data path/to/NIBSS_FRAUD_DATASET.csv

This takes approximately 5-10 minutes on a standard laptop.
It saves models/rf_model.pkl and models/iso_forest.pkl
compatible with your local Python/sklearn version.
"""

import argparse, sys, os, time, warnings, pickle, json
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Train and save XAI fraud detection models.')
parser.add_argument('--data', required=True, help='Path to NIBSS_FRAUD_DATASET.csv')
args = parser.parse_args()

if not os.path.exists(args.data):
    print(f"ERROR: File not found: {args.data}")
    sys.exit(1)

print("="*60)
print("XAI Fraud Detection — Model Setup")
print("="*60)

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import ADASYN
import xgboost as xgb

SEED = 42
np.random.seed(SEED)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\n[1/7] Loading dataset from {args.data}...")
t0 = time.time()
df = pd.read_csv(args.data, low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
print(f"      {len(df):,} rows loaded in {time.time()-t0:.1f}s")

print("[2/7] Time-aware train-test split (Jan-Oct 2023 / Nov-Dec 2023)...")
train = df[df['timestamp'] < '2023-11-01'].copy()
test  = df[df['timestamp'] >= '2023-11-01'].copy()
print(f"      Train: {len(train):,} rows | Test: {len(test):,} rows")

CAT  = ['channel','merchant_category','bank','location','age_group']
DROP = ['transaction_id','customer_id','timestamp','is_fraud','fraud_technique']
FEAT = [c for c in df.columns if c not in DROP]
tr_enc = pd.get_dummies(train[FEAT], columns=CAT)
te_enc = pd.get_dummies(test[FEAT],  columns=CAT)
tr_enc, te_enc = tr_enc.align(te_enc, join='left', axis=1, fill_value=0)
FNAMES = list(tr_enc.columns)
y_tr = train['is_fraud'].values
y_te = test['is_fraud'].values

print("[3/7] Creating stratified training sample (all fraud + 80,000 legit)...")
fi = np.where(y_tr==1)[0]; li = np.where(y_tr==0)[0]
rng = np.random.default_rng(SEED)
sl  = rng.choice(li, size=80000, replace=False)
idx = np.concatenate([fi, sl]); rng.shuffle(idx)
Xs  = tr_enc.values.astype(float)[idx]
ys  = y_tr[idx]
X_te = te_enc.values.astype(float)
print(f"      Sample: {len(ys):,} rows ({ys.sum():,} fraud)")

print("[4/7] Fitting StandardScaler and Isolation Forest...")
sc = StandardScaler()
Xs_sc  = sc.fit_transform(Xs)
Xte_sc = sc.transform(X_te)

iso = IsolationForest(n_estimators=100, contamination=0.003,
                      random_state=SEED, n_jobs=-1)
iso.fit(Xs_sc)
if_tr = iso.score_samples(Xs_sc).reshape(-1,1)
if_te = iso.score_samples(Xte_sc).reshape(-1,1)
Xs_aug  = np.hstack([Xs_sc,  if_tr])
Xte_aug = np.hstack([Xte_sc, if_te])
FNAMES_AUG = FNAMES + ['if_anomaly_score']

print("[5/7] Applying ADASYN oversampling (training only)...")
ada = ADASYN(sampling_strategy=0.20, n_neighbors=5, random_state=SEED)
Xr, yr = ada.fit_resample(Xs_aug, ys)
print(f"      After ADASYN: {len(yr):,} rows ({yr.sum():,} fraud)")

print("[6/7] Training Random Forest (n_estimators=200)... [~3-5 mins]")
t1 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=20,
     class_weight='balanced_subsample', random_state=SEED, n_jobs=-1)
rf.fit(Xr, yr)
print(f"      Done in {time.time()-t1:.0f}s")

print("[7/7] Training XGBoost and saving all models...")
xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
    eval_metric='aucpr', n_jobs=-1, verbosity=0)
xgb_m.fit(Xr, yr)

# Quick validation
p_rf  = rf.predict_proba(Xte_aug)[:,1]
p_xgb = xgb_m.predict_proba(Xte_aug)[:,1]
p_ens = (p_rf + p_xgb) / 2
prauc = average_precision_score(y_te, p_xgb)
print(f"\n      Validation XGB PR-AUC: {prauc:.4f} (expected ~0.8037)")

# Save models
with open(f'{MODEL_DIR}/rf_model.pkl','wb')    as f: pickle.dump(rf, f)
with open(f'{MODEL_DIR}/iso_forest.pkl','wb')  as f: pickle.dump(iso, f)
with open(f'{MODEL_DIR}/scaler.pkl','wb')      as f: pickle.dump(sc, f)
xgb_m.save_model(f'{MODEL_DIR}/xgb_model.json')

# Save scaler as JSON backup
scaler_params = {
    'mean_': sc.mean_.tolist(), 'scale_': sc.scale_.tolist(),
    'var_': sc.var_.tolist(),   'n_features_in_': int(sc.n_features_in_),
    'n_samples_seen_': int(sc.n_samples_seen_),
}
with open(f'{MODEL_DIR}/scaler_params.json','w') as f: json.dump(scaler_params, f)
with open(f'{MODEL_DIR}/feature_names.json','w') as f: json.dump(FNAMES_AUG, f)

# Save feature importance
fi_vals  = rf.feature_importances_
feat_imp = sorted(zip(FNAMES_AUG, fi_vals.tolist()), key=lambda x: -x[1])
results_path = f'{MODEL_DIR}/results.json'
with open(results_path) as f: results_data = json.load(f)
results_data['feat_imp'] = feat_imp[:15]
with open(results_path,'w') as f: json.dump(results_data, f, indent=2)

print("\n" + "="*60)
print("Setup complete! Model files saved to models/")
for fn in sorted(os.listdir(MODEL_DIR)):
    sz = os.path.getsize(f'{MODEL_DIR}/{fn}')
    print(f"  {fn}: {sz//1024}KB")
print("\nRun the app with: streamlit run app.py")
print("="*60)
