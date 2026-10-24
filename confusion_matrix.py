
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

# =====================================
# 1. 데이터셋 불러오기
# =====================================

data = np.load(
    "dataset_trial_split.npz",
    allow_pickle=True
)

Xtr = data["Xtr"]
ytr = data["ytr"]
Xte = data["Xte"]
yte = data["yte"]

print("===== 데이터셋 확인 =====")
print("학습 데이터:", Xtr.shape)
print("테스트 데이터:", Xte.shape)

# =====================================
# 2. 라벨 숫자 변환
# =====================================

le = LabelEncoder()

ytr = le.fit_transform(ytr)
yte = le.transform(yte)

classes = le.classes_
n_classes = len(classes)

print("\n클래스:", classes)

# =====================================
# 3. 모델 정의
# =====================================

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

# =====================================
# 4. 결과 저장 폴더
# =====================================

os.makedirs("images", exist_ok=True)

# =====================================
# 5. 모델별 학습 및 평가
# =====================================

for name, model in models.items():

    print(f"\n===== {name} =====")
    print("모델 학습 중...", flush=True)

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)

    acc = accuracy_score(yte, pred)
    precision = precision_score(
        yte, pred,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        yte, pred,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        yte, pred,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")

    # =================================
    # 6. Confusion Matrix 계산
    # =================================

    cm = confusion_matrix(
        yte,
        pred,
        labels=np.arange(n_classes)
    )

    print("\nConfusion Matrix:")
    print(cm)

    # =================================
    # 7. 클래스별 분석
    # =================================

    print("\n===== 클래스별 분석 =====")

    row_sums = cm.sum(axis=1)
    correct = np.diag(cm)

    class_accuracy = np.divide(
        correct,
        row_sums,
        out=np.zeros(n_classes, dtype=float),
        where=row_sums != 0
    )

    for i, label in enumerate(classes):
        print(
            f"{label}: "
            f"{correct[i]}/{row_sums[i]} 정답 "
            f"({class_accuracy[i] * 100:.2f}%)"
        )

    best_idx = np.argmax(class_accuracy)
    worst_idx = np.argmin(class_accuracy)

    print("\n가장 잘 분류된 클래스:", classes[best_idx])
    print("가장 낮은 정답률 클래스:", classes[worst_idx])

    # 가장 많이 혼동한 실제 클래스 → 예측 클래스
    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    i, j = np.unravel_index(
        np.argmax(errors),
        errors.shape
    )

    if errors[i, j] > 0:
        print(
            "가장 많이 발생한 오분류:",
            f"{classes[i]} → {classes[j]}",
            f"({errors[i, j]}개)"
        )
    else:
        print("오분류가 발생하지 않았습니다.")

    # =================================
    # 8. Confusion Matrix 그림
    # =================================

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    plt.colorbar(im, ax=ax)

    threshold = cm.max() / 2

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9
            )

    plt.tight_layout()

    filename = f"images/confusion_matrix_{name.lower()}.png"
    plt.savefig(filename, dpi=200)
    plt.close()

    print("\n이미지 저장:", filename)

print("\n===== 모든 모델 분석 완료 =====")
