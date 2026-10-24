
import glob
import os
import numpy as np

from collections import Counter
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# =====================================
# 1. 기본 설정
# =====================================

FS = 1000
LEN = 2000
WIN = 200
HOP = 100

STARTS = {
    "1.5~3.5초": 1500,
    "1.0~3.0초": 1000
}

files = sorted(glob.glob("data/**/*.csv", recursive=True))

if not files:
    raise ValueError("CSV 파일을 찾을 수 없습니다.")

def label_of(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split("_", 1)[1].replace(" ", "_")

# =====================================
# 2. 필터링
# =====================================

def preprocess(x):
    bn, an = iirnotch(60, 30, FS)
    x = filtfilt(bn, an, x, axis=0)

    b, a = butter(
        4,
        [20 / (FS / 2), 450 / (FS / 2)],
        btype="band"
    )

    return filtfilt(b, a, x, axis=0)

# =====================================
# 3. 시간 영역 특징
# =====================================

def td_features(w, th=0.01):
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    mav = np.mean(np.abs(w), axis=0)

    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)

    zc = np.sum(
        ((w[:-1] * w[1:]) < 0)
        & (np.abs(d) >= th),
        axis=0
    )

    ssc = np.sum(
        (d[:-1] * d[1:]) < 0,
        axis=0
    )

    return np.concatenate([rms, mav, wl, zc, ssc])

# =====================================
# 4. 원본 CSV에서 시행 추출
# =====================================

trials = []
labels = []
trial_ids = []

print("===== 시행 추출 중 =====")

for file_idx, path in enumerate(files):
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1
    )

    marks = np.where(data[:, 7] == 1)[0]

    for i in range(len(marks) - 1):
        trial = data[marks[i]:marks[i + 1], :5]

        if len(trial) < max(STARTS.values()) + LEN:
            continue

        trials.append(trial)
        labels.append(label_of(path))
        trial_ids.append((file_idx, i))

labels = np.asarray(labels)
indices = np.arange(len(trials))

print("전체 시행:", len(trials))
print("클래스별 시행:", Counter(labels))

# =====================================
# 5. 시행 단위로 학습/검증 분할
# =====================================

# 기존 시행 단위 실험과 동일한 외부 분할
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 외부 테스트 시행은 이번 구간 비교에 사용하지 않음
# 남은 학습 시행에서 내부 검증용 시행 분할
fit_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.2,
    stratify=labels[train_idx],
    random_state=42
)

assert set(fit_idx).isdisjoint(set(val_idx))
assert set(fit_idx).isdisjoint(set(test_idx))
assert set(val_idx).isdisjoint(set(test_idx))

print("\n===== 시행 분할 확인 =====")
print("모델 학습 시행:", len(fit_idx))
print("검증 시행:", len(val_idx))
print("보관된 테스트 시행:", len(test_idx))
print("학습/검증 중복:", len(set(fit_idx) & set(val_idx)))

# =====================================
# 6. 구간별 특징 생성
# =====================================

def build(selected_indices, start):
    X, y = [], []

    for idx in selected_indices:
        # 시행별 필터링
        xf = preprocess(trials[idx])

        seg = xf[start:start + LEN]

        for pos in range(0, LEN - WIN + 1, HOP):
            wnd = seg[pos:pos + WIN]
            X.append(td_features(wnd))
            y.append(labels[idx])

    return np.asarray(X), np.asarray(y)

# =====================================
# 7. 두 구간 비교
# =====================================

print("\n===== 분석 구간별 검증 성능 =====")

for name, start in STARTS.items():
    print(f"\n{name} 특징 생성 중...", flush=True)

    X_fit, y_fit = build(fit_idx, start)
    X_val, y_val = build(val_idx, start)

    # 표준화는 모델 학습 데이터로만 학습
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_val = scaler.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_fit, y_fit)
    pred = model.predict(X_val)

    acc = accuracy_score(y_val, pred)
    f1 = f1_score(
        y_val, pred,
        average="macro",
        zero_division=0
    )

    print("학습 특징:", X_fit.shape)
    print("검증 특징:", X_val.shape)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1: {f1 * 100:.2f}%")

print("\n===== 비교 완료 =====")
print("외부 테스트 데이터는 사용하지 않았습니다.")
