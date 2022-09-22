using System;
namespace PinballController
{
    [Flags]
    public enum PinballControls
    {
        LeftFlipper = 0b00000001,
        RightFlipper = 0b00000010,
        Plunger = 0b00000100,
        StartButton = 0b00001000
    };
}
