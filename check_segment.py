
import glob
import os
import numpy as np

from collections import Counter
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================
# 1. 기본 설정
# =====================================

FS = 1000
START = 1000
LEN = 2000
WIN = 200
HOP = 100

files = sorted(glob.glob("data/**/*.csv", recursive=True))

if not files:
    raise ValueError("CSV 파일을 찾을 수 없습니다.")

# =====================================
# 2. 클래스 라벨 추출
# =====================================

def label_of(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split("_", 1)[1].replace(" ", "_")

# =====================================
# 3. 필터링
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
# 4. 슬라이딩 윈도우
# =====================================

def make_windows(x):
    n = (len(x) - WIN) // HOP + 1

    return np.stack([
        x[i * HOP:i * HOP + WIN]
        for i in range(n)
    ])

# =====================================
# 5. 시간 영역 특징 추출
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

    return np.concatenate([
        rms, mav, wl, zc, ssc
    ])

# =====================================
# 6. CSV에서 시행 추출
# =====================================

trials = []
labels = []
trial_ids = []

print("===== 시행 추출 시작 =====")

for file_idx, path in enumerate(files):
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1
    )

    marks = np.where(data[:, 7] == 1)[0]

    # 각 시행을 독립적으로 저장
    for i in range(len(marks) - 1):
        trial = data[marks[i]:marks[i + 1], :5]

        if len(trial) < START + LEN:
            continue

        trials.append(trial)
        labels.append(label_of(path))
        trial_ids.append((file_idx, i))

print("전체 시행 수:", len(trials))
print("클래스별 시행 수:", Counter(labels))

# =====================================
# 7. 시행 단위 학습/테스트 분할
# =====================================

indices = np.arange(len(trials))

tr_idx, te_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 같은 시행이 양쪽에 없는지 검증
assert set(tr_idx).isdisjoint(set(te_idx))

print("\n===== 시행 단위 분할 =====")
print("학습 시행:", len(tr_idx))
print("테스트 시행:", len(te_idx))
print("시행 중복:", len(set(tr_idx) & set(te_idx)))

# =====================================
# 8. 분할 후 특징 생성
# =====================================

def build(selected_indices):
    X, y = [], []

    for idx in selected_indices:
        trial = trials[idx]

        # 각 시행별 필터 적용
        xf = preprocess(trial)

        seg = xf[START:START + LEN]

        for wnd in make_windows(seg):
            X.append(td_features(wnd))
            y.append(labels[idx])

    return np.asarray(X), np.asarray(y)

print("\n학습 데이터 생성 중...")
Xtr, ytr = build(tr_idx)

print("테스트 데이터 생성 중...")
Xte, yte = build(te_idx)

# =====================================
# 9. 표준화
# =====================================

scaler = StandardScaler()

Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

# =====================================
# 10. 결과 확인 및 저장
# =====================================

print("\n===== 최종 데이터셋 =====")
print("학습:", Xtr.shape, ytr.shape)
print("테스트:", Xte.shape, yte.shape)

np.savez(
    "dataset_segment_1000.npz",
    Xtr=Xtr,
    ytr=ytr,
    Xte=Xte,
    yte=yte
)

print("\ndataset_segment_1000.npz 저장 완료!")