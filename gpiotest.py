import Jetson.GPIO as GPIO

def main():
    print("Hello World!")
    print(' GPIO info ', GPIO.JETSON_INFO)
    print(' GPIO info ', GPIO.VERSION)
    
    GPIO.setmode(GPIO.BOARD)
    mode = GPIO.getmode()
    GPIO.setwarnings(False)
    print('mode ', mode)
    
    channels = [33, 29]
    GPIO.setup(channels, GPIO.OUT)
    print('Channel ', GPIO.gpio_function(29))
    GPIO.output(29, GPIO.LOW)
    GPIO.cleanup()
    
if __name__ == "__main__":
    main()
