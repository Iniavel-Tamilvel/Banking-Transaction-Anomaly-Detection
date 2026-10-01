import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/sample_transactions.csv")
features=["amount","frequency_7d","account_age_days","merchant_risk","location_distance_km","hour","is_international"]
print(df.shape); print(df.isna().sum())
sns.scatterplot(data=df,x="location_distance_km",y="amount",hue="is_international",size="merchant_risk",sizes=(30,200))
plt.title("Transaction Amount vs Location Distance"); plt.tight_layout(); plt.savefig(ROOT/"visualisations/amount_vs_distance.png",dpi=180); plt.close()
X=StandardScaler().fit_transform(df[features])
model=IsolationForest(n_estimators=300,contamination=.15,random_state=42)
df["anomaly_label"]=model.fit_predict(X); df["anomaly_score"]=model.decision_function(X)
print(df.sort_values("anomaly_score")[["transaction_id","amount","anomaly_score","anomaly_label"]].head())
