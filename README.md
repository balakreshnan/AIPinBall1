# AIPinBall1

- Nvidia Xavier Dev Kit
- Install Jetpack 5.02
- Install Torch for AML models - https://forums.developer.nvidia.com/t/torch-cuda-is-available-returns-false/183821/3
- This is will torch 1.12 which works and also enables GPU
- pip install json_tricks
- pip install objdict

## For AML Trained Model

- python onnxvideoaml.py
- Make sure the video capture is set to appropriate device id for our test it was 0

## For Azure Cognitive Service Custom vision Tensorflow export Trained Model

- python onnxvideo.py
- Make sure the video capture is set to appropriate device id for our test it was 0
