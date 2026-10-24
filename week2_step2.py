
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt

FS = 1000
START = 1500
LEN = 2000

# 1. CSV 파일 불러오기
file = 'data/a_30 fist.csv'

data = np.loadtxt(
    file,
    delimiter=',',
    skiprows=1
)

# 실제 sEMG 5채널
x = data[:, :5]

# 2. 필터 함수 (1단계와 동일)
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

# 4. mark를 기준으로 첫 번째 시행 추출
marks = np.where(data[:, 7] == 1)[0]

if len(marks) < 2:
    raise ValueError("시행 경계가 부족합니다.")

trial = xf[marks[0]:marks[1]]

# 5. 2초 분석 구간 추출
def cut(x, start=START, n=LEN):
    return x[start:start+n]

seg = cut(trial)

print("===== 분석 구간 추출 결과 =====")
print("파일:", file)
print("시행 모양:", trial.shape)
print("세그먼트 모양:", seg.shape)

if seg.shape != (2000, 5):
    raise ValueError("세그먼트 모양이 (2000, 5)가 아닙니다.")

# 6. 추출한 구간을 그래프로 확인
t = np.arange(len(trial)) / FS

fig, ax = plt.subplots(
    5, 1,
    figsize=(11, 8),
    sharex=True
)

for ch in range(5):
    ax[ch].plot(
        t,
        trial[:, ch],
        linewidth=0.5
    )

    # 추출한 구간 표시
    ax[ch].axvspan(
        START / FS,
        (START + LEN) / FS,
        color='orange',
        alpha=0.2
    )

    ax[ch].axvline(
        START / FS,
        color='red',
        linestyle='--'
    )

    ax[ch].axvline(
        (START + LEN) / FS,
        color='red',
        linestyle='--'
    )

    ax[ch].set_ylabel(f'CH{ch+1}')
    ax[ch].grid(alpha=0.3)

ax[4].set_xlabel('Time (s)')

fig.suptitle(
    '2-Second Analysis Segment (30 deg Fist)',
    fontsize=13
)

plt.tight_layout()
plt.savefig('week2_step2_segment.png', dpi=150)
plt.show()

print("그래프 저장 완료: week2_step2_segment.png")
