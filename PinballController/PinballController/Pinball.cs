using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO.Ports;
using System.Runtime.InteropServices;
using RGiesecke.DllExport;
namespace PinballController
{


    public class Pinball: IPinball
    {

        static bool _continue;
        static SerialPort _serialPort;
        private System.Threading.Thread _thread;

        public event EventHandler DataReceived;

        protected virtual void OnDataReceived(EventArgs e)
        {
            EventHandler handler = DataReceived;
            handler?.Invoke(this, e);
        }

        public event EventHandler DataSent;

        protected virtual void OnDataSent(EventArgs e)
        {
            EventHandler handler = DataSent;
            handler?.Invoke(this, e);
        }

        public event EventHandler Disconnected;
        protected virtual void OnDisconnected(EventArgs e)
        {
            EventHandler handler = Disconnected;
            handler?.Invoke(this, e);
        }


        public class DataReceivedEventArgs : EventArgs
        {
            //public string data { get; set; }
            public PinballControls controls;
        }

        public class DataSendEventArgs : EventArgs
        {
            public string data { get; set; }
        }

        public Pinball(string port)
        {



            // Create a new SerialPort object with default settings.
            _serialPort = new SerialPort();

            // Allow the user to set the appropriate properties.
            _serialPort.PortName = port;
            _serialPort.BaudRate = 115200;
            _serialPort.Parity = Parity.None;
            _serialPort.DataBits = 8;
            _serialPort.StopBits = StopBits.One;
            _serialPort.Handshake = Handshake.None;

            // Set the read/write timeouts
            _serialPort.ReadTimeout = 500;
            _serialPort.WriteTimeout = 500;


        }

        ~Pinball()
        {
            Disconnect();
        }

        private string port;
        /// <summary>
        /// Gets or sets the serial port.  Expects a string text value, such as "COM5".
        /// </summary>
        public string Port
        {
            get { return _serialPort.PortName; }
            set
            {
                port = value;
                if (_serialPort.IsOpen) _serialPort.Close();
                _serialPort.PortName = value;
                //EventArgs e = new DataReceivedEventArgs();
                //OnDisconnected(e);
                Disconnected?.Invoke(this, EventArgs.Empty);
            }
        }

        private int baud;
        /// <summary>
        /// Gets or sets the baud rate.
        /// </summary>
        public int BaudRate
        {
            get { return _serialPort.BaudRate; }
            set { baud = value; _serialPort.BaudRate = baud; }
        }


        /// <summary>
        /// Turns on the channel (ControlPush) and automatically turns off the channel (ControlRelease) after a pre-programmed amount of time.
        /// Also called triggered.
        /// </summary>
        /// <param name="channel"></param>
        [DllExport("ControlPushAndRelease", CallingConvention = CallingConvention.Cdecl)]
        public void ControlPushAndRelease(PinballControls mask)
        {

            byte accum1 = 0b10010000;   // 9

            if (mask.HasFlag(PinballControls.LeftFlipper)) accum1 = (byte)(accum1 | 0b00000001);
            if (mask.HasFlag(PinballControls.RightFlipper)) accum1 = (byte)(accum1 | 0b00000010);
            if (mask.HasFlag(PinballControls.Plunger)) accum1 = (byte)(accum1 | 0b00000100);
            if (mask.HasFlag(PinballControls.StartButton)) accum1 = (byte)(accum1 | 0b00001000);

            SendSerialCommand(accum1);
        }

        /// <summary>
        /// Turns on a specific channel (without changing the state of any other channel).
        /// </summary>
        /// <param name="channel"></param>
        public void ControlPush(PinballControls mask)
        {

            byte accum1 = 0b10100000;   // 10

            if (mask.HasFlag(PinballControls.LeftFlipper)) accum1 = (byte)(accum1 | 0b00000001);
            if (mask.HasFlag(PinballControls.RightFlipper)) accum1 = (byte)(accum1 | 0b00000010);
            if (mask.HasFlag(PinballControls.Plunger)) accum1 = (byte)(accum1 | 0b00000100);
            if (mask.HasFlag(PinballControls.StartButton)) accum1 = (byte)(accum1 | 0b00001000);

            SendSerialCommand(accum1);
        }

        /// <summary>
        /// Turns off a specific channel (without changing the state of any other channel).
        /// </summary>
        /// <param name="channel"></param>
        public void ControlRelease(PinballControls mask)
        {
            byte accum1 = 0b10110000; // 11
            if (mask.HasFlag(PinballControls.LeftFlipper)) accum1 = (byte)(accum1 | 0b00000001);
            if (mask.HasFlag(PinballControls.RightFlipper)) accum1 = (byte)(accum1 | 0b00000010);
            if (mask.HasFlag(PinballControls.Plunger)) accum1 = (byte)(accum1 | 0b00000100);
            if (mask.HasFlag(PinballControls.StartButton)) accum1 = (byte)(accum1 | 0b00001000);

            SendSerialCommand(accum1);
        }


        /// <summary>
        /// Connects to the serial port.  Assumes an Arduino with the Pinball sketch is loaded.  
        /// </summary>
        /// <returns></returns>
        /// 
        [DllExport("Connect", CallingConvention = CallingConvention.Cdecl)]
        public bool Connect()
        {
            bool rc = true;
            try
            {
                _serialPort.Open();
                _continue = true;
                //readThread.Start();
                _thread = new System.Threading.Thread(Read);
                _thread.Start();

                Console.WriteLine("Connected!");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Connecton failed!" + ex.Message);
                rc = false;
            }

            return rc;
        }

        /// <summary>
        /// Disconnects the serial port.
        /// </summary>
        public void Disconnect()
        {
            try
            {
                if (_serialPort != null)
                {
                    if (_serialPort.IsOpen)
                    {
                        // Send command to shutoff all relays then close.
                        _serialPort.WriteLine("@00"); // Turn off all relays.
                        // Wait a second
                        System.Threading.Thread.Sleep(1000);
                        _serialPort.Close();

                    }

                }
                _continue = false;

                Console.WriteLine("Disconnected!");
            }
            catch (Exception)
            {
                Console.WriteLine("Disconnect failed!");
            }

        }

        /// <summary>
        /// This method "latches" or sets the state of each channel by setting each bit of the low low_nibble of a byte.
        /// For safety reasons, the command is ignored if you attempt to override the high low_nibble as this will
        /// override the command protocol.
        /// </summary>
        /// <param name="value"></param>
        public void SetLatchState(byte low_nibble)
        {
            if (low_nibble <= 0xF)
            {
                byte data = (byte)(0b10000000 | low_nibble);  //8
                SendSerialCommand(data);
            }

        }

        /// <summary>
        /// Internal command for sending a single byte of data to the serial port.  The data is defined as:
        /// high nibble = Which command to execute on the Arduino
        /// low nibble = the 4 bits representing each channel on the pinball interface.
        /// bin    int   description
        /// ===============================================
        /// 1000   8     Latch   (all 4 relays)
        /// 1001   9     Trigger (specified relays) - Acts like a mometary press.  The Arduino will release automatically about 100 milliseconds.
        /// 1010   10    Hold    (specified relays)
        /// 1011   11    Release (specified relays)
        /// 1100   12    [unused]
        /// 1101   13    [unused]
        /// 1110   14    [unused]
        /// 1111   15    [unused]
        /// </summary>
        /// <param name="value"></param>
        private void SendSerialCommand(byte value)
        {
            byte[] buff = new byte[1] { value };
            if (_serialPort.IsOpen)
                _serialPort.Write(buff, 0, 1); // write a single byte!

            // Raise event
            DataSendEventArgs e = new DataSendEventArgs();
            e.data = value.ToString();
            OnDataSent(e);
        }


        /// <summary>
        /// This method is executed on a separate thread and automatically processes any serial data sent back from the Arduino.
        /// </summary>
        private void Read()
        {
            while (_continue)
            {
                try
                {
                    if (_serialPort.IsOpen)
                    {
                        if (_serialPort.BytesToRead > 0)
                        {
                            byte data = (byte)_serialPort.ReadByte();

                            Console.WriteLine(data.ToString());
                            DataReceivedEventArgs e = new DataReceivedEventArgs();
                            e.controls = (PinballControls)data;
                            OnDataReceived(e);

                        }

                    }

                }
                catch (TimeoutException)
                {
                    // Do nothing
                }
                catch (System.IO.IOException)
                {
                    // Do nothing
                }
            }
        }




    }
}
