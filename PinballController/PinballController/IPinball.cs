using System;
using System.Runtime.InteropServices;
using System.EnterpriseServices;
namespace PinballController
{
    public interface IPinball
    {
        int BaudRate { get; set; }
        string Port { get; set; }

        event EventHandler DataReceived;
        event EventHandler DataSent;
        event EventHandler Disconnected;

        [DispId(0)]
        bool Connect();

        [DispId(1)]
        void ControlPush(PinballControls mask);
        [DispId(2)]
        void ControlRelease(PinballControls mask);
        [DispId(3)]
        void ControlPushAndRelease(PinballControls mask);

        [DispId(3)]
        void Disconnect();
        //void SetLatchState(byte low_nibble);
    }
}