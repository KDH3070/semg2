
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

# 2. 필터 함수
def preprocess(x, fs=FS):
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)

    b, a = butter(
        4,
        [20/(fs/2), 450/(fs/2)],
        btype='band'
    )

    return filtfilt(b, a, x, axis=0)

# 3. 필터 적용
xf = preprocess(x)

# 4. 첫 번째 시행 추출
marks = np.where(data[:, 7] == 1)[0]

if len(marks) < 2:
    raise ValueError("시행 경계를 확인할 수 없습니다.")

trial = xf[marks[0]:marks[1]]

# 5. 2초 분석 구간 추출
def cut(x, start=START, n=LEN):
    return x[start:start+n]

seg = cut(trial)

if seg.shape != (LEN, 5):
    raise ValueError("세그먼트 모양이 올바르지 않습니다.")

# 6. 슬라이딩 윈도우 함수
def make_windows(x, win=WIN, hop=HOP):
    n = (len(x) - win) // hop + 1

    return np.stack([
        x[i*hop:i*hop+win]
        for i in range(n)
    ])

# 7. 윈도우 생성
w = make_windows(seg)

# 8. 결과 출력
print("===== 슬라이딩 윈도우 결과 =====")
print("파일:", file)
print("세그먼트 모양:", seg.shape)
print("윈도우 배열:", w.shape)
print("윈도우 개수:", len(w))
print("윈도우 길이:", w.shape[1])
print("채널 수:", w.shape[2])

# 9. 결과 검증
assert w.shape == (19, 200, 5), \
    "윈도우 배열 모양이 예상과 다릅니다."

print("\n슬라이딩 윈도우 생성 성공!")
