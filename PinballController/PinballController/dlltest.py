import ctypes

pbc = ctypes.CDLL("C:\\Users\\AiPinBot\\AIPinBall1\\PinballController\\PinballController\\bin\\x86\\Release\\net471\\PinballController.dll")
pbc.Port = "COM3"
pbc.Connect()
