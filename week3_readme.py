
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Step2에서 생성한 데이터셋 불러오기
d = np.load("dataset_trial_split.npz", allow_pickle=True)

Xtr = d["Xtr"]
ytr = d["ytr"]
Xte = d["Xte"]
yte = d["yte"]

print("===== 데이터셋 확인 =====")
print("학습 데이터:", Xtr.shape)
print("테스트 데이터:", Xte.shape)

# 2. 문자열 클래스 라벨을 숫자로 변환
le = LabelEncoder()
ytr = le.fit_transform(ytr)
yte = le.transform(yte)

print("\n클래스:", le.classes_)

# 3. 세 모델 정의
models = {
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        decision_function_shape="ovr"
    ),
    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "XGB": XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# 4. 모델 학습
print("\n===== 모델 학습 시작 =====")

for name, model in models.items():
    print(f"\n{name} 학습 중...", flush=True)
    model.fit(Xtr, ytr)
    print(f"{name} 학습 완료!")

print("\n===== 세 모델 학습 완료 =====")


# 5. 모델별 성능 평가
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("\n===== 모델별 성능 비교 =====")
print(f"{'Model':<10} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'F1-score':>10}")

results = {}

for name, model in models.items():
    pred = model.predict(Xte)

    acc = accuracy_score(yte, pred)
    precision = precision_score(yte, pred, average="macro", zero_division=0)
    recall = recall_score(yte, pred, average="macro", zero_division=0)
    f1 = f1_score(yte, pred, average="macro", zero_division=0)

    results[name] = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

    print(
        f"{name:<10} "
        f"{acc*100:>9.2f}% "
        f"{precision*100:>10.2f}% "
        f"{recall*100:>9.2f}% "
        f"{f1*100:>9.2f}%"
    )
