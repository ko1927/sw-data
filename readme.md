<br>
<br>

# 문제제기
제공된 데이터는 test(50000개), train(55000개), unlabel(1200개) 3종류입니다.<br>
트랜스포머, cnn, mlp 등 모델 학습을 통해 val데이터에 대한 정확도 99%달성 하였습니다.<br>
하지만 test 데이터와 유형이 달라 정확도가 낮아지는 현상이 존재합니다.<br>
데이터 분석부터 보여드리겠습니다.
<br>
<br>

# 데이터를 보기전 문제보기

- 5초 분량의 입력 오디오 샘플에서 영어 음성의 진짜(Real) 사람의 목소리와 생성 AI의 가짜(Fake) 사람의 목소리를 동시에 검출해내는 AI 모델을 개발해야합니다.
- 학습 데이터는 방음 환경에서 녹음된 진짜(Real) 사람의 목소리 샘플과 방음 환경을 가정한 가짜(Fake) 사람의 목소리로 구성되어 있으며, ***각 샘플 당 사람의 목소리는 1개*** 입니다.
- 평가 데이터는 5초 분량의 다양한 환경에서의 오디오 샘플로 구성되며, **샘플 당 최대 2개의 진짜(Real) 혹은 가짜(Fake) 사람의 목소리가 동시에 존재**합니다.
- **Unlabel 데이터는 학습에 활용할 수 있지만 Label이 제공되지 않으며, 평가 데이터의 환경과 동일합니다.**
<br>
<br>
<br>
<br>

# 데이터 분석
## 학습 데이터와 평가 데이터의 간극

### 1. 종류의 차이
**학습데이터는 한개 + 방음. 평가는 두개 + 소음입니다**

```
학습 데이터 : 각 샘플 당 사람의 목소리는 1개
평가 데이터 : 샘플 당 최대 2개의 진짜(Real) 혹은 가짜(Fake) 사람의 목소리가 동시에 존재, 하나만 존재할 수 있음
```


### 2. 시각화 비교
아래의 데이터는 훈련 및 테스트 데이터의 *특징추출을 시각화* 한 사진입니다.<br>
두 데이터는 각 train, test데이터를 mel스펙트럼으로 변환 저장한 것입니다. 
<br>
해당 데이터를 학습하고 분석하여 검출하게 됩니다. 데이터들은 길이나 간격, 잡음 등 차이가 존재합니다.
- 훈련데이터 샘플 AAACWKPZ<br>
![alt text](AAACWKPZ.png)
- 테스트데이터 샘플 TEST_00000<br>
![alt text](TEST_00000.png)

### 3. 소음데이터
평가 데이터 중 소음에 가려진 데이터가 많았습니다.<br>
아래 데이터가 바로 그 대표적인 예시입니다. 검은 색이 없이 대부분 소음으로 채워져 있습니다. 이러한 소리가 섞인 것이 오차범위의 데이터라고 추정하고 있습니다.
- 사진 TEST_00021<br>
![alt text](TEST_00021-1.png)

- 음성 TEST_00021<br>
[TEST_00021.ogg](TEST_00021.ogg).
<br>


### 4. 학습당시의 잡음
아래의 소리를 들어보시면 잡음이 있음을 알게 됩니다. 마이크 자체의 소음일 수 있으며, 현재는 그대로 학습에 사용하고 있습니다.
- 사진 AHJQEQQY<br>
![alt text](AHJQEQQY.png)
- 음성 AHJQEQQY<br>
[AHJQEQQY.ogg](AHJQEQQY.ogg).

### 5. unlabel의 분석
**Unlabel 데이터는 학습에 활용할 수 있지만 Label이 제공되지 않으며, 평가 데이터의 환경과 동일합니다.** 
다음은 그 스펙트럼입니다.
<br>
평가 데이터와 닮아 있음을 알 수 있습니다.
<br>
어떻게 쓸지는 마음대로이나 저희는 domain적응을 활용하고자 합니다.
- 사진 BCFCIBHG<br>
![alt text](BCFCIBHG.png)
- 사진 ADUNDKGO<br>
![alt text](ADUNDKGO.png)
- 사진 BELZRKJI<br>
![alt text](BELZRKJI.png)
- 사진 CKDVTZGL<br>
![alt text](CKDVTZGL.png)

# 모델 학습 및 분석

## 기본 MLP 학습


###  1. 모델
기본 MLP모델 사용, 음원 자체 파일 특징 추출로 학습

```python
class MLP(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=128, output_dim=CONFIG.N_CLASSES):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)
```

### 2. 평가
val 데이터에 대한 정확도 98.93% <br>
평가데이터 오율 : 0.4618
```
	id	path	label	예측
0	PUOXNOKJ	./train/PUOXNOKJ.ogg	real	real
1	GXOIPDJP	./train/GXOIPDJP.ogg	fake	fake
2	FOEQKPPR	./train/FOEQKPPR.ogg	fake	fake
3	IYASAVDT	./train/IYASAVDT.ogg	real	real
4	VLWIXPTC	./train/VLWIXPTC.ogg	real	real
...	...	...	...	...
11083	WQMWFZRS	./train/WQMWFZRS.ogg	fake	fake
11084	KYLYAJSQ	./train/KYLYAJSQ.ogg	fake	fake
11085	AEFBUARF	./train/AEFBUARF.ogg	real	real
11086	VDPZMHZX	./train/VDPZMHZX.ogg	fake	fake
11087	NAEMKYCJ	./train/NAEMKYCJ.ogg	real	real
```

## CNN모델 학습

사진을 변환하고 이걸 그냥 불러와서 학습하자는 아이디어
### 1. 데이터 변환 음성 -> 사진
```python
def save_spectrogram_image(y, sr, out_path, n_mels=128, hop_length=256):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()

def convert_audio_folder_to_spectrogram(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.ogg') or f.endswith('.wav')]

    for audio_file in audio_files:
        input_path = os.path.join(input_folder, audio_file)
        output_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + '.png')

        y, sr = librosa.load(input_path, sr=CONFIG.SR)
        save_spectrogram_image(y, sr, output_path)
```
결과 <br>

- AHJQEQQY.ogg -> AHJQEQQY.png
![alt text](AHJQEQQY.png)<br>

### 2. CNN 모델

```python
class CNN(nn.Module):
    def __init__(self, output_dim=CONFIG.N_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)

```


### 3. 평가

val 데이터에 대한 정확도 99.93% <br>
평가데이터 오율 : 0.4186

## MLP + Pretrain, 파인튜닝

### 