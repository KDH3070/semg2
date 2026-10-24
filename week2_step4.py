
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

FS = 1000
START = 1500
LEN = 2000

WIN = 200
HOP = 100

# 1. CSV 파일 불러오기
file = 'data/a_30 fist.csv'

data = np.loadtxt(
    file,
    delimiter=',',
    skiprows=1
)

# 실제 sEMG 5채널만 선택
x = data[:, :5]


# 2. 필터 적용
def preprocess(x, fs=FS):
    # 60Hz 노치 필터
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)

    # 20~450Hz 대역통과 필터
    b, a = butter(
        4,
        [20/(fs/2), 450/(fs/2)],
        btype='band'
    )

    return filtfilt(b, a, x, axis=0)


xf = preprocess(x)


# 3. 첫 번째 시행 추출
marks = np.where(data[:, 7] == 1)[0]

if len(marks) < 2:
    raise ValueError("시행 경계를 찾을 수 없습니다.")

trial = xf[marks[0]:marks[1]]


# 4. 2초 세그먼트 추출
seg = trial[START:START+LEN]

if seg.shape != (2000, 5):
    raise ValueError("세그먼트 모양이 올바르지 않습니다.")


# 5. 슬라이딩 윈도우
def make_windows(x, win=WIN, hop=HOP):
    n = (len(x) - win) // hop + 1

    return np.stack([
        x[i*hop:i*hop+win]
        for i in range(n)
    ])


w = make_windows(seg)


# 6. 시간영역 특징 추출
def td_features(w, th=0.01):
    # w: (200, 5)

    # RMS
    rms = np.sqrt(np.mean(w**2, axis=0))

    # MAV
    mav = np.mean(np.abs(w), axis=0)

    # WL
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)

    # ZC
    zc = np.sum(
        ((w[:-1] * w[1:]) < 0) &
        (np.abs(d) >= th),
        axis=0
    )

    # SSC
    ssc = np.sum(
        ((d[:-1] * d[1:]) < 0),
        axis=0
    )

    return np.concatenate([
        rms, mav, wl, zc, ssc
    ])


# 7. 첫 번째 윈도우 특징 추출
f = td_features(w[0])


# 8. 결과 출력
print("===== 시간영역 특징 추출 결과 =====")
print("파일:", file)
print("세그먼트 모양:", seg.shape)
print("윈도우 배열:", w.shape)
print("특징 벡터:", f.shape)

print("\n특징 값:")
print(f)


# 9. 결과 검증
assert f.shape == (25,), \
    "특징 벡터가 25차원이 아닙니다."

if not np.all(np.isfinite(f)):
    raise ValueError("특징 벡터에 NaN 또는 무한대가 있습니다.")

print("\n25차원 특징 추출 성공!")


# 10. ZC 임계값 비교 (추가 기능)
f_zero = td_features(w[0], th=0)

print("\n===== ZC 임계값 비교 =====")
print("th=0.01:", f[15:20])
print("th=0   :", f_zero[15:20])

print("\nZC 임계값 비교 완료!")
