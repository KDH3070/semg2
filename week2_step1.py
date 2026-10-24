
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

FS = 1000

# 1. CSV 파일 불러오기
file = 'data/a_30 fist.csv'

data = np.loadtxt(
    file,
    delimiter=',',
    skiprows=1
)

# 2. 실제 sEMG 5채널만 선택
# bandpass 1~5
x = data[:, :5]

# 3. 필터 함수
def preprocess(x, fs=FS):

    # 60Hz 노치 필터
    bn, an = iirnotch(60, 30, fs)

    x = filtfilt(
        bn, an, x,
        axis=0
    )

    # 20~450Hz 대역통과 필터
    b, a = butter(
        4,
        [20/(fs/2), 450/(fs/2)],
        btype='band'
    )

    return filtfilt(
        b, a, x,
        axis=0
    )

# 4. 필터 적용
xf = preprocess(x)

# 5. 필터 전후 표준편차 비교
print("===== 필터 적용 결과 =====")
print("파일:", file)
print("원본 배열:", x.shape)
print("필터 후 배열:", xf.shape)

print("\n채널별 표준편차")
print("채널      필터 전       필터 후")

for ch in range(5):
    print(
        f"CH{ch+1}     "
        f"{x[:, ch].std():.4f}       "
        f"{xf[:, ch].std():.4f}"
    )

print("\n전체 표준편차")
print("전:", x.std().round(4))
print("후:", xf.std().round(4))
