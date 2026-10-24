
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

FS = 1000
WIN = 200
HOP = 100

files = sorted(glob.glob("data/**/*.csv", recursive=True))

if not files:
    raise ValueError("CSV 파일을 찾을 수 없습니다.")

# 파일명에서 클래스 이름 추출
def label_of(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split("_", 1)[1].replace(" ", "_")

# 클래스별 대표 CSV 파일 하나씩 선택
class_files = {}

for path in files:
    label = label_of(path)

    if label not in class_files:
        class_files[label] = path

print("===== 클래스별 확인 파일 =====")
for label, path in sorted(class_files.items()):
    print(label, ":", path)

# 시행별 RMS 계산
def calculate_rms(trial):
    values = []

    for start in range(0, len(trial) - WIN + 1, HOP):
        window = trial[start:start + WIN]

        rms = np.sqrt(np.mean(window ** 2, axis=0))
        values.append(np.mean(rms))

    return np.array(values)

# 8개 클래스 그래프
fig, axes = plt.subplots(
    4, 2,
    figsize=(14, 12),
    sharex=True
)

axes = axes.flatten()

for ax, (label, path) in zip(
    axes, sorted(class_files.items())
):
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1
    )

    marks = np.where(data[:, 7] == 1)[0]

    rms_list = []

    # 첫 5개 완전한 시행
    for i in range(min(5, len(marks) - 1)):
        trial = data[marks[i]:marks[i + 1], :5]
        rms_list.append(calculate_rms(trial))

    rms_list = np.array(rms_list)

    time = (
        np.arange(rms_list.shape[1]) * HOP + WIN / 2
    ) / FS

    for rms in rms_list:
        ax.plot(time, rms, alpha=0.25)

    ax.plot(
        time,
        rms_list.mean(axis=0),
        color="black",
        linewidth=2
    )

    ax.axvspan(
        1.5, 3.5,
        color="orange",
        alpha=0.15
    )

    ax.axvspan(
        2.0, 3.0,
        color="green",
        alpha=0.15
    )

    ax.set_title(label)
    ax.set_ylabel("Mean RMS")
    ax.grid(alpha=0.2)

for ax in axes[-2:]:
    ax.set_xlabel("Time (s)")

plt.suptitle(
    "RMS Comparison Across 8 Classes",
    fontsize=15
)

plt.tight_layout()
plt.show()
