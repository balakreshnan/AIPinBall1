import RPi.GPIO as GPIO
import time

# Pin Definitions
output_pin = 29  # BCM pin 18, BOARD pin 12

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

    print("Starting demo now! Press CTRL+C to exit")
    print(' GPIO info ', GPIO.JETSON_INFO)
    print(' GPIO info ', GPIO.VERSION)
    curr_value = GPIO.HIGH
    try:
        while True:
            time.sleep(5)
            print('Input value ', GPIO.input(output_pin))
            # Toggle the output every second
            print("Outputting {} to pin {}".format(curr_value, output_pin))
            GPIO.output(output_pin, curr_value)
            curr_value ^= GPIO.HIGH
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    main()
