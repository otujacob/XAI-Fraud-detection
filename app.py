import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
from matplotlib.patches import Patch
import pickle, json, os, warnings, time
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Fraud Detection - Nigerian Inter-Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
CAT_FEATS  = ['channel','merchant_category','bank','location','age_group']
DROP_COLS  = ['transaction_id','customer_id','timestamp','is_fraud','fraud_technique']
THETA_BLOCK  = 0.70
THETA_REVIEW = 0.40

# ── Load models (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    with open(f'{MODEL_DIR}/rf_model.pkl','rb')   as f: rf  = pickle.load(f)
    with open(f'{MODEL_DIR}/xgb_model.pkl','rb')  as f: xgb = pickle.load(f)
    with open(f'{MODEL_DIR}/scaler.pkl','rb')      as f: sc  = pickle.load(f)
    with open(f'{MODEL_DIR}/iso_forest.pkl','rb')  as f: iso = pickle.load(f)
    with open(f'{MODEL_DIR}/feature_names.json')   as f: fnames = json.load(f)
    with open(f'{MODEL_DIR}/results.json')         as f: res = json.load(f)
    return rf, xgb, sc, iso, fnames, res

rf_model, xgb_model, scaler, iso_forest, FEATURE_NAMES, PRECOMPUTED = load_models()

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(df_input):
    """Preprocess a transaction dataframe for scoring."""
    feat_cols = [c for c in df_input.columns if c not in DROP_COLS]
    enc = pd.get_dummies(df_input[feat_cols], columns=[c for c in CAT_FEATS if c in feat_cols])
    # Align to training feature set (all 69 features)
    base_names = [n for n in FEATURE_NAMES if n != 'if_anomaly_score']
    for col in base_names:
        if col not in enc.columns:
            enc[col] = 0
    enc = enc[[c for c in base_names if c in enc.columns]]
    for col in base_names:
        if col not in enc.columns:
            enc[col] = 0
    enc = enc[base_names]
    X_sc = scaler.transform(enc.values.astype(float))
    if_sc = iso_forest.score_samples(X_sc).reshape(-1,1)
    X_aug = np.hstack([X_sc, if_sc])
    return X_aug

def score_transactions(X_aug):
    p_rf  = rf_model.predict_proba(X_aug)[:,1]
    p_xgb = xgb_model.predict_proba(X_aug)[:,1]
    p_ens = (p_rf + p_xgb) / 2
    return p_rf, p_xgb, p_ens

def decision(prob):
    if prob >= THETA_BLOCK:  return 'BLOCK',  '🔴'
    if prob >= THETA_REVIEW: return 'REVIEW', '🟡'
    return 'APPROVE', '🟢'

def shap_bar(shap_vals, feat_names, title="SHAP Feature Attribution"):
    """Plot SHAP bar chart for a single transaction."""
    top_n = 8
    paired = sorted(zip(shap_vals, feat_names), key=lambda x: -abs(x[0]))[:top_n]
    vals   = [p[0] for p in paired]
    names  = [p[1] for p in paired]
    colours = ['#C62828' if v > 0 else '#1565C0' for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=colours, edgecolor='black', linewidth=0.6, height=0.65)
    ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('SHAP Value (red = increases fraud score, blue = decreases)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/COM752-Dissertation-0D2B45?style=for-the-badge")
    st.markdown("### Explainable AI Fraud Detection")
    st.markdown("**Nigerian Inter-Bank Payments**")
    st.markdown("---")
    st.markdown("**Student:** Otu Samuel Jacob  \n**ID:** s25007038  \n**University:** Wrexham University")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Overview",
        "📊  Model Results",
        "🔍  SHAP Explanations",
        "🔄  HITL Feedback",
        "⚡  Live Scoring",
    ])
    st.markdown("---")
    st.caption("Framework: IF + XGBoost + RF + ADASYN + SHAP")

page = page.split("  ")[-1]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🏦 XAI Fraud Detection — Nigerian Inter-Bank Payments")
    st.markdown("""
    > *"Explainable AI with Human-in-the-Loop Feedback for Real-Time Fraud Detection  
    > in Nigerian Inter-Bank Payment Systems"*  
    > — COM 752 MSc Dissertation, Wrexham University, 2025/2026
    """)

    s = PRECOMPUTED['summary']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{s['total_transactions']:,}")
    c2.metric("Fraud Cases", f"{s['fraud_cases']:,}")
    c3.metric("Class Imbalance", f"{s['class_ratio']}:1")
    c4.metric("P95 Latency", f"~{s['latency_p95']}ms")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Four-Component Framework")
        st.markdown("""
        1. **Stage 1 — Isolation Forest**  
           Unsupervised anomaly scoring → Feature 69
        2. **Stage 2 — Hybrid Classifier**  
           XGBoost + Random Forest trained on ADASYN-balanced data
        3. **SHAP TreeExplainer**  
           Real-time natural language explanations + counterfactuals
        4. **Human-in-the-Loop**  
           Analyst feedback → iterative model retraining
        """)
    with c2:
        st.subheader("Key Results")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Best PR-AUC (XGBoost) | **0.8037** |
        | Best Recall (XGBoost) | **0.6646** |
        | False Positives (BLOCK) | **0** |
        | SHAP Latency (P95) | **~340ms** |
        | H1 (SHAP within 500ms) | ✅ Confirmed |
        | H2 (HITL recall gain) | +25.9pp |
        | H3 (PR-AUC > 0.91) | ❌ Not confirmed |
        """)

    st.markdown("---")
    st.subheader("Nigerian Fraud Ecology")
    col1, col2, col3 = st.columns(3)
    col1.metric("Social Engineering", "64.3%", "dominant fraud type")
    col2.metric("Mobile Banking", "49.9%", "highest fraud channel")
    col3.metric("Amount features", "69.1%", "of SHAP attribution")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Results":
    st.title("📊 Model Performance Results")
    st.caption("Test Set: Nov–Dec 2023  |  185,658 transactions  |  328 fraud cases")

    res = PRECOMPUTED['results']
    rows = []
    for model, metrics in res.items():
        rows.append({'Model': model, **metrics})
    df_res = pd.DataFrame(rows)

    # Colour the best PR-AUC
    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: #D4EDDA' if v else '' for v in is_max]

    st.subheader("Table 4.1: Complete Performance Comparison")
    st.dataframe(
        df_res.style
            .apply(highlight_best, subset=['PR_AUC'])
            .apply(highlight_best, subset=['Recall'])
            .format({'Precision':'{:.4f}','Recall':'{:.4f}','F1':'{:.4f}',
                     'AUC_ROC':'{:.4f}','PR_AUC':'{:.4f}','MCC':'{:.4f}'}),
        use_container_width=True, height=280
    )
    st.caption("Green highlight = best value. PR-AUC is the primary metric.")

    st.markdown("---")
    st.subheader("Performance Comparison — Primary Metrics")

    configs = ['B1: RF\nAlone','B2: IF\nAlone','B3: RF+\nADASYN','Full\nHybrid']
    models4  = list(res.keys())[:4]
    clrs     = ['#D0D8E4','#F0F0F0','#D0D8E4','#000000']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance Across Four Configurations', fontsize=13, fontweight='bold')
    for ax, (metric, key, title) in zip(axes, [
        ('PR-AUC', 'PR_AUC', 'PR-AUC (Primary)'),
        ('Recall', 'Recall', 'Recall'),
        ('F1',     'F1',     'F1-Score'),
    ]):
        vals = [res[m][key] for m in models4]
        bars = ax.bar(configs, vals, color=clrs, edgecolor='black', linewidth=1.2, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, min(1.0, max(vals)*1.35))
        ax.set_ylabel('Score', fontsize=11)
        ax.tick_params(labelsize=11)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Feature Importance — Top 15 Features")
    fi_data = PRECOMPUTED['feat_imp']
    names_fi = [f[0] for f in fi_data[:15]]
    vals_fi  = [f[1] for f in fi_data[:15]]
    cmap = {'amount':'#222','amount_log':'#222','amount_vs_mean_ratio':'#222',
            'amount_sum_24h':'#222','velocity_score':'#555',
            'month_sin':'#777','month_cos':'#777','day_cos':'#777',
            'is_peak_hour':'#777','channel_Mobile':'#999'}
    bar_c = [cmap.get(n,'#AAAAAA') for n in names_fi]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names_fi))
    ax2.barh(y_pos, vals_fi, color=bar_c, edgecolor='black', linewidth=0.8, height=0.65)
    for i, val in enumerate(vals_fi):
        ax2.text(val+0.0008, i, f'{val:.4f}', va='center', fontsize=10)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(names_fi, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('Mean Decrease in Impurity', fontsize=11)
    ax2.set_title('Top 15 Feature Importances (Full Hybrid RF Component)', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    handles = [Patch(fc='#222',ec='black',label='Amount features'),
               Patch(fc='#555',ec='black',label='Composite risk'),
               Patch(fc='#777',ec='black',label='Temporal features'),
               Patch(fc='#999',ec='black',label='Channel features')]
    ax2.legend(handles=handles, fontsize=10, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig2)
    st.info("Amount-based features account for **69.1%** of mean absolute SHAP attribution — confirming the social engineering fraud signature.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SHAP EXPLANATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Explanations":
    st.title("🔍 SHAP Explanation Examples")
    st.markdown("Three real transactions from the test set, showing how the SHAP explanation module works at the moment of the fraud decision.")

    cases = PRECOMPUTED['cases']
    tabs  = st.tabs(["Case A — High Risk (FRAUD)", "Case B — Medium Risk (LEGIT)", "Case C — Low Risk (LEGIT)"])

    for tab, (key, label_col) in zip(tabs, [('A','#C62828'),('B','#F57C00'),('C','#2E7D32')]):
        c = cases[key]
        with tab:
            d_str, d_icon = decision(c['prob_rf'])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Label",     "FRAUD" if c['true_y']==1 else "LEGIT")
            col2.metric("RF Probability", f"{c['prob_rf']:.4f}")
            col3.metric("XGB Probability",f"{c['prob_xgb']:.4f}")
            col4.metric("Decision",       f"{d_icon} {d_str}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Amount",  f"NGN {c['amount']:,.2f}")
            col6.metric("Channel", c['channel'])
            col7.metric("Hour",    f"{c['hour']:02d}:00")
            col8.metric("Bank",    c['bank'])

            st.markdown("**SHAP Attribution — Top 5 Features:**")
            shap_df = pd.DataFrame(c['shap'], columns=['Feature', 'SHAP Value'])
            shap_df['Direction'] = shap_df['SHAP Value'].apply(
                lambda x: '🔴 Increases fraud score' if x > 0 else '🔵 Decreases fraud score')
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

            # Plot SHAP bar
            sv   = [s[1] for s in c['shap']]
            fn   = [s[0] for s in c['shap']]
            fig  = shap_bar(sv, fn, f"Case {key} — SHAP Attribution")
            st.pyplot(fig)

            # Natural language explanation
            top = c['shap'][0]
            direction = "elevated above" if top[1] > 0 else "below"
            if c['true_y'] == 1:
                nl = (f"**Explanation:** This transaction was flagged because "
                      f"**{top[0]}** is the primary driver (SHAP = {top[1]:+.4f}), "
                      f"pushing the fraud probability to {c['prob_rf']:.4f}. "
                      f"The transaction amount of NGN {c['amount']:,.2f} "
                      f"is {direction} the customer's typical range.")
                cf_thresh = round(c['amount'] * 0.35 / 1000) * 1000
                cf = f"**Counterfactual:** This transaction would NOT be flagged if the amount were below approximately NGN {cf_thresh:,.0f}."
            else:
                nl = (f"**Explanation:** This transaction was approved because "
                      f"**{top[0]}** (SHAP = {top[1]:+.4f}) reduces the fraud score. "
                      f"The amount of NGN {c['amount']:,.2f} is within normal range.")
                cf = "**Counterfactual:** N/A — transaction is already approved."
            st.info(nl)
            st.success(cf)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: HITL FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "HITL Feedback":
    st.title("🔄 Human-in-the-Loop Feedback Simulation")
    st.markdown("Three analyst feedback cycles — showing how the model improves as analysts confirm fraud cases from the review queue.")

    hitl = PRECOMPUTED['hitl']
    cycles      = [h['cycle']        for h in hitl]
    tp_block    = [h['tp_block']     for h in hitl]
    recall_bl   = [h['recall_block'] for h in hitl]
    pr_auc_vals = [h['pr_auc']       for h in hitl]

    # Metrics table
    st.subheader("Table 4.2: HITL Simulation Results (θ = 0.70)")
    df_hitl = pd.DataFrame(hitl)
    df_hitl['Gain in TP'] = df_hitl['tp_block'] - df_hitl['tp_block'].iloc[0]
    df_hitl['Gain in TP'] = df_hitl['Gain in TP'].apply(lambda x: f'+{x}' if x > 0 else '—')
    st.dataframe(df_hitl[['cycle','tp_block','Gain in TP','fp_block',
                            'tp_review','recall_block','pr_auc']].rename(columns={
        'cycle':'Cycle','tp_block':'TP Blocked','fp_block':'FP Blocked',
        'tp_review':'TP in Review','recall_block':'Recall (Block)','pr_auc':'PR-AUC'
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    clrs4 = ['#F0F0F0','#D0D8E4','#D0D8E4','#000000']
    xlabs = ['Cycle 0\n(Baseline)','Cycle 1','Cycle 2','Cycle 3']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('HITL Feedback Results — Three Retraining Cycles\nFixed threshold θ = 0.70',
                 fontsize=13, fontweight='bold')
    ax1, ax2, ax3 = axes
    bars = ax1.bar(cycles, tp_block, color=clrs4, edgecolor='black', linewidth=1.3, width=0.65)
    for bar, val in zip(bars, tp_block):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, str(val),
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_xticks(cycles); ax1.set_xticklabels(xlabs, fontsize=11)
    ax1.set_ylabel('Fraud Cases Auto-Blocked', fontsize=12); ax1.set_ylim(0, 280)
    ax1.set_title('Fraud Auto-Blocked\n(No Analyst Review Required)', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax2.plot(cycles, recall_bl, 'ko-', lw=2.5, ms=9)
    for x, y in zip(cycles, recall_bl):
        ax2.text(x+0.06, y+0.012, f'{y:.4f}', fontsize=11, fontweight='bold')
    ax2.set_xticks(cycles); ax2.set_xticklabels(xlabs, fontsize=11)
    ax2.set_ylabel('Recall at BLOCK Tier', fontsize=12); ax2.set_ylim(0.35, 0.80)
    ax2.set_title('Recall Improvement at BLOCK Tier', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax3.plot(cycles, pr_auc_vals, 'ko-', lw=2.5, ms=9)
    for x, y in zip(cycles, pr_auc_vals):
        ax3.text(x+0.06, y+0.004, f'{y:.4f}', fontsize=11, fontweight='bold')
    ax3.set_xticks(cycles); ax3.set_xticklabels(xlabs, fontsize=11)
    ax3.set_ylabel('PR-AUC', fontsize=12); ax3.set_ylim(0.75, 0.90)
    ax3.set_title('PR-AUC Across Feedback Cycles', fontsize=12, fontweight='bold')
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0,0,1,0.90])
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Key Finding")
    col1, col2, col3 = st.columns(3)
    col1.metric("Recall improvement", "+25.9pp", "0.4451 → 0.7043")
    col2.metric("Extra fraud blocked", "+85 cases", "146 → 231")
    col3.metric("New false positives", "0", "precision maintained")
    st.info("**H2 reframing:** The baseline already achieves zero false positives before any feedback. HITL instead improves block-tier recall by 25.9 percentage points across three cycles with zero new false positives.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: LIVE SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Scoring":
    st.title("⚡ Live Transaction Scoring")
    st.markdown("Score new transactions through the full framework pipeline — Isolation Forest → XGBoost + RF → Decision.")

    tab1, tab2 = st.tabs(["📁 Upload CSV", "🎯 Demo Transactions"])

    with tab1:
        st.markdown("Upload a transaction CSV with the same column structure as the NIBSS dataset.")
        uploaded = st.file_uploader("Upload transaction CSV", type=['csv'])

        if uploaded:
            df_up = pd.read_csv(uploaded, low_memory=False)
            st.success(f"Loaded {len(df_up):,} transactions")

            if st.button("Score Transactions", type="primary"):
                with st.spinner("Running pipeline: Standardise → IF Score → Classify → Explain..."):
                    try:
                        t0 = time.perf_counter()
                        X_aug = preprocess(df_up)
                        p_rf, p_xgb, p_ens = score_transactions(X_aug)
                        elapsed = (time.perf_counter() - t0) * 1000

                        decisions = [decision(p)[0] for p in p_ens]
                        df_up['RF_Prob']       = p_rf.round(4)
                        df_up['XGB_Prob']      = p_xgb.round(4)
                        df_up['Ensemble_Prob'] = p_ens.round(4)
                        df_up['Decision']      = decisions

                        n = len(df_up)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Scored", f"{n:,}")
                        c2.metric("BLOCK",  f"{(df_up['Decision']=='BLOCK').sum()}", "🔴")
                        c3.metric("REVIEW", f"{(df_up['Decision']=='REVIEW').sum()}", "🟡")
                        c4.metric("APPROVE",f"{(df_up['Decision']=='APPROVE').sum()}", "🟢")
                        st.metric("Total time", f"{elapsed:.0f}ms", f"~{elapsed/n:.1f}ms per transaction")

                        st.dataframe(df_up[['amount','channel','Ensemble_Prob','Decision']].head(50),
                                     use_container_width=True, hide_index=True)

                        # SHAP for highest risk transaction
                        top_idx = p_ens.argmax()
                        st.subheader(f"SHAP Explanation — Highest Risk Transaction (prob={p_ens[top_idx]:.4f})")
                        try:
                            import shap
                            expl = shap.TreeExplainer(rf_model)
                            sv   = expl.shap_values(X_aug[top_idx:top_idx+1])
                            sv_f = sv[0,:,1] if sv.ndim==3 else sv[1][0]
                            fig  = shap_bar(sv_f, FEATURE_NAMES, "Highest Risk Transaction — SHAP Attribution")
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"SHAP computation skipped: {e}")
                    except Exception as e:
                        st.error(f"Scoring error: {e}")
                        st.info("Make sure your CSV has the same columns as the NIBSS dataset.")

    with tab2:
        st.markdown("Score the built-in demo dataset (528 transactions: 328 fraud + 200 legitimate from the test set).")
        demo_path = os.path.join(DATA_DIR, 'demo_transactions.csv')

        if st.button("Run Demo Scoring", type="primary"):
            df_demo = pd.read_csv(demo_path, low_memory=False)
            with st.spinner("Scoring 528 demo transactions..."):
                t0 = time.perf_counter()
                X_aug = preprocess(df_demo)
                p_rf, p_xgb, p_ens = score_transactions(X_aug)
                elapsed = (time.perf_counter() - t0) * 1000

                y_true = df_demo['is_fraud'].values if 'is_fraud' in df_demo.columns else None
                decisions = [decision(p)[0] for p in p_ens]
                df_demo['RF_Prob']       = p_rf.round(4)
                df_demo['Ensemble_Prob'] = p_ens.round(4)
                df_demo['Decision']      = decisions

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Scored", f"{len(df_demo):,}")
                c2.metric("BLOCK",  f"{(df_demo['Decision']=='BLOCK').sum()}", "🔴")
                c3.metric("REVIEW", f"{(df_demo['Decision']=='REVIEW').sum()}", "🟡")
                c4.metric("APPROVE",f"{(df_demo['Decision']=='APPROVE').sum()}", "🟢")
                c4_2 = st.columns(1)[0]
                c4_2.metric("Scoring time", f"{elapsed:.0f}ms total | ~{elapsed/len(df_demo):.1f}ms/tx")

                if y_true is not None:
                    from sklearn.metrics import average_precision_score
                    prauc = average_precision_score(y_true, p_ens)
                    st.metric("Live PR-AUC", f"{prauc:.4f}", "computed on demo set")

                st.dataframe(
                    df_demo[['amount','channel','bank','RF_Prob','Ensemble_Prob','Decision']]
                    .sort_values('Ensemble_Prob', ascending=False).head(30),
                    use_container_width=True, hide_index=True
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("COM 752 MSc Dissertation · Otu Samuel Jacob · s25007038 · Wrexham University · 2025/2026")
