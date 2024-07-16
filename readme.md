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
대회 링크 : https://dacon.io/competitions/official/236253/overview/description
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
어떻게 쓸지는 마음대로이나 저희는 domain적응에 활용하고자 합니다.
- 사진 BCFCIBHG<br>
![alt text](BCFCIBHG.png)
- 사진 ADUNDKGO<br>
![alt text](ADUNDKGO.png)
- 사진 BELZRKJI<br>
![alt text](BELZRKJI.png)
- 사진 CKDVTZGL<br>
![alt text](CKDVTZGL.png)

# 모델 학습 및 분석
점수 산정 기준 : Score = 0.5 × (1 − AUC) + 0.25 × Brier + 0.25 × ECE

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
val에 대한 학습은 잘 되어있다.

## CNN모델 학습

사진을 변환하고 이걸 그냥 불러와서 학습하자는 아이디어를 기반하여 테스트
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
unlabel 데이터를 먼저 넣어서 auto인코더를 진행하고 이후 train데이터 기반 학습 진행<br>
pretrain 후 파인튜닝하였고, 사용데이터는 음성

### 1. autoencoder
```python
# Autoencoder for pretraining
class Autoencoder(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

### 2. train데이터 MLP 
```python
class PretrainedMLP(nn.Module):
    def __init__(self, autoencoder, input_dim=CONFIG.N_MFCC, hidden_dim=128, output_dim=CONFIG.N_CLASSES):
        super(PretrainedMLP, self).__init__()
        self.encoder = autoencoder.encoder
        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)
```

### 3. 평가

val 데이터에 대한 정확도 98.9% <br>
평가데이터 오율 : 50회 학습의 경우 0.4574, 5회 학습의 경우 0.4391

## CNN + 의사 라벨링

### 1. 모델
```python
class CNNModel(nn.Module):
    def __init__(self, n_mels, time_steps):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * (n_mels // 8) * (time_steps // 8), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: 'real' and 'fake'

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 2. 의사 라벨링
```python
def pseudo_labeling(model, unlabeled_loader):
    model.eval()
    pseudo_labels = []
    with torch.no_grad():
        for inputs in unlabeled_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            pseudo_labels.extend(predicted.cpu().numpy())
    return np.array(pseudo_labels)
```

### 3. 평가
val 데이터에 대한 정확도 99.76% <br>
평가데이터 오율 : 0.4711

## MLP + 의사라벨링

### 1. 모델
```python
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

### 2. 의사라벨링
```python
with torch.no_grad():
    for inputs in tqdm(unlabeled_loader, desc="Generating Pseudo-labels"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        pseudo_labels.extend(predicted.cpu().numpy())

pseudo_labels = np.array(pseudo_labels)
```

### 3. 평가
val 데이터에 대한 정확도 99.79% <br>
평가데이터 오율 : 0.4187

## vision_transformer
변환한 사진을 vision_transformer로 학습

### 1. 학습의 4가지 경우
```python
#vit_b_16
trained_model, train_loader, val_loader, test_loader = train_model('vit_b_16', ViT_B_32_Weights.IMAGENET1K_V1)

#vit_b_32
trained_model, train_loader, val_loader, test_loader = train_model('vit_b_32', ViT_B_32_Weights.IMAGENET1K_V1)

#vit_l_16
trained_model, train_loader, val_loader, test_loader = train_model('vit_l_16', ViT_L_16_Weights.IMAGENET1K_V1)

#vit_l_32
trained_model, train_loader, val_loader, test_loader = train_model('vit_l_32', ViT_L_32_Weights.IMAGENET1K_V1)
```

### 2. 평가
모델들의 정답률 자체는 같으나, 오답에 대한 %보정이 달라 다음과 같은 결과가 발생
<br>

- vit_b_16
```
평가데이터 오율 : 0.4727
```
- vit_b_32
```
평가데이터 오율 : 0.4652
```
- vit_l_16
```
평가데이터 오율 : 0.4724
```
- vit_l_32
```
평가데이터 오율 : 0.4143
```


# 전체 요약

## 모델평가
- 데이터 자체의 학습은 되어있어 평가률은 대부분 90%가 넘는다.
- test 데이터에 대해서는 대부분 0.41~45정도의 오율을 가진다.

## 원인 분석
- 데이터의 domain차이 및 오염으로 모델이 구분하지 못하는 현상이 발생한다고 추정. 
- TEST 데이터는 2개의 소리, 다양한 환경이나 TRAIN은 순수한 소리 1개라 차이가 있다고 추정
- 실제 데이터의 스펙트럼 사진만 보아도 차이를 알 수 있음.
- 정제된 표지판 학습 후 오염된 사진을 구분하지 못하는 유사해보이는 경우도 찾음.
<br>

![alt text](<스크린샷 2024-07-17 005421.png>)
<br>
- unlabel데이터를 활용하여 극복하여야 한다는 아이디어 구상

## 시도한 방법
- 적대적 학습 : 0.4857 오율이 나와 현재는 중단하고 다음 방법을 찾는 중
- pretrain + 파인튜닝 : 데이터의 차이 극복을 어느정도 하였으나 미미함
- 의사 라벨링 : cnn모델은 오율이 높았으나 MLP의 경우 오율이 낮았음. 여전히 미미함
- 모델 변경 : MLP, CNN, Transformer 모델을 사용하였으며 모델 중 정답률이 가장 높은 것은 Transformer모델이었음. 현재는 Transformer모델을 활용해 개선하고자 함.