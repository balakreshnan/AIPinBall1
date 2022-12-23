# AIPinBall1

- Nvidia Xavier Dev Kit
- Install Jetpack 5.02
- Install Torch for AML models - https://forums.developer.nvidia.com/t/torch-cuda-is-available-returns-false/183821/3
- This is will torch 1.12 which works and also enables GPU
- pip install json_tricks
- pip install objdict
- sudo pip install jetson-stats
- pip install openai
- pip install gym

## For AML Trained Model

- python onnxvideoaml.py
- Make sure the video capture is set to appropriate device id for our test it was 0

## For Azure Cognitive Service Custom vision Tensorflow export Trained Model

- python onnxvideo.py
- Make sure the video capture is set to appropriate device id for our test it was 0

## Refinforcement learning need tensorflow environment

```
python3 -m virtualenv -p python tfenv
```

```
source tfenv/bin/activate
```

- install dependencies inside env

```
pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta setuptools testresources
```

- #sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v502 tensorflow==2.10.0+nv22.11

```
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow==2.10.0+nv22.11
```
