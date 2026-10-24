
import glob
import os
import numpy as np
from collections import Counter
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.model_selection import train_test_split

FS = 1000
START = 1500
LEN = 2000
WIN = 200
HOP = 100

# 1. CSV 파일 목록
files = sorted(glob.glob('data/**/*.csv', recursive=True))

if not files:
    raise ValueError("CSV 파일을 찾을 수 없습니다.")


# 2. 파일명에서 클래스 추출
def label_of(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split('_', 1)[1].replace(' ', '_')


labels = [label_of(f) for f in files]


# 3. 파일 단위 학습/테스트 분할
tr_f, te_f = train_test_split(
    files,
    test_size=0.2,
    stratify=labels,
    random_state=42
)


# 4. 필터 함수
def preprocess(x, fs=FS):
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)

    b, a = butter(
        4,
        [20/(fs/2), 450/(fs/2)],
        btype='band'
    )

    return filtfilt(b, a, x, axis=0)


# 5. 슬라이딩 윈도우
def make_windows(x, win=WIN, hop=HOP):
    n = (len(x) - win) // hop + 1

    return np.stack([
        x[i*hop:i*hop+win]
        for i in range(n)
    ])


# 6. 시간영역 특징
def td_features(w, th=0.01):
    rms = np.sqrt(np.mean(w**2, axis=0))
    mav = np.mean(np.abs(w), axis=0)

    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)

    zc = np.sum(
        ((w[:-1] * w[1:]) < 0) &
        (np.abs(d) >= th),
        axis=0
    )

    ssc = np.sum(
        ((d[:-1] * d[1:]) < 0),
        axis=0
    )

    return np.concatenate([rms, mav, wl, zc, ssc])


# 7. 각 집합에서만 특징 생성
def build(file_list):
    X, y = [], []

    for p in file_list:
        data = np.loadtxt(
            p,
            delimiter=',',
            skiprows=1
        )

        # sEMG 5채널
        x = data[:, :5]

        # 필터 적용
        xf = preprocess(x)

        # mark 기준 시행 경계
        marks = np.where(data[:, 7] == 1)[0]

        # 완전한 시행만 사용
        for i in range(len(marks) - 1):
            trial = xf[marks[i]:marks[i+1]]

            seg = trial[START:START+LEN]

            if seg.shape != (LEN, 5):
                continue

            for wnd in make_windows(seg):
                X.append(td_features(wnd))
                y.append(label_of(p))

    return np.asarray(X), np.asarray(y)


# 8. 학습/테스트 특징 생성
print("===== 파일 단위 분할 =====")
print("전체 파일:", len(files))
print("학습 파일:", len(tr_f))
print("테스트 파일:", len(te_f))

print("\n학습 파일 클래스 분포:")
print(dict(sorted(Counter(label_of(p) for p in tr_f).items())))

print("\n테스트 파일 클래스 분포:")
print(dict(sorted(Counter(label_of(p) for p in te_f).items())))

print("\n학습 데이터 특징 생성 중...")
Xtr, ytr = build(tr_f)

print("테스트 데이터 특징 생성 중...")
Xte, yte = build(te_f)

# 9. 결과 확인
print("\n===== 데이터셋 생성 결과 =====")
print("Xtr:", Xtr.shape)
print("ytr:", ytr.shape)
print("Xte:", Xte.shape)
print("yte:", yte.shape)

assert Xtr.shape[1] == 25
assert Xte.shape[1] == 25
assert set(tr_f).isdisjoint(set(te_f))
assert np.all(np.isfinite(Xtr))
assert np.all(np.isfinite(Xte))

print("\n파일 단위 데이터 분할 및 특징 추출 성공!")



from sklearn.preprocessing import StandardScaler

# 10. 학습 데이터 기준으로 표준화
sc = StandardScaler()

Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)

# 11. 표준화 결과 확인
print("\n===== 표준화 결과 =====")

print("학습셋 평균:")
print(Xtr_s.mean(axis=0).round(6))

print("\n학습셋 표준편차:")
print(Xtr_s.std(axis=0).round(6))

print("\n테스트셋 평균:")
print(Xte_s.mean(axis=0).round(6))

# 12. 데이터셋 저장
np.savez(
    'dataset.npz',
    Xtr=Xtr_s,
    ytr=ytr,
    Xte=Xte_s,
    yte=yte
)

print("\n===== 저장 결과 =====")
print("학습 데이터:", Xtr_s.shape)
print("테스트 데이터:", Xte_s.shape)
print("저장 완료: dataset.npz")
