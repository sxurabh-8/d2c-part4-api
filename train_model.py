"""Re-trains the churn model from data/rfm_modeling_snapshot.csv and writes model.pkl.
Run only if model.pkl is missing or you want to retrain."""
import joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

DATA = Path(__file__).resolve().parent.parent / "data" / "rfm_modeling_snapshot.csv"
OUT  = Path(__file__).resolve().parent / "model.pkl"

def main():
    df = pd.read_csv(DATA)
    target = "churn_next_60d"; split = "split"
    drop = ["customer_id", "snapshot_date", target, split]
    feat = [c for c in df.columns if c not in drop]
    cat = df[feat].select_dtypes(include="object").columns.tolist()
    num = [c for c in feat if c not in cat]
    df[cat] = df[cat].fillna("Unknown"); df[num] = df[num].fillna(0)
    tr = df[df[split]=="train"]; va = df[df[split]=="validation"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    pipe = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42))])
    pipe.fit(tr[feat], tr[target])
    vp = pipe.predict_proba(va[feat])[:,1]
    print("val AUC:", roc_auc_score(va[target], vp))
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.2, 0.71, 0.02):
        f1 = precision_recall_fscore_support(va[target], (vp>=t).astype(int), average="binary", zero_division=0)[2]
        if f1 > best_f1: best_t, best_f1 = float(t), float(f1)
    print("threshold:", best_t)
    joblib.dump({"model": pipe, "threshold": best_t, "feature_cols": feat, "cat_cols": cat, "num_cols": num}, OUT)
    print("Saved", OUT)

if __name__ == "__main__":
    main()
