# Binary Decoder for CAEN events from V1740B digitizer obtained with the board reader

Run the binary decoder with

`python decoder.py -N 1 -P 0`

For getting more information about other options use

`python decoder.py -h`

```
usage: decoder.py [-h] -N N -P {0,1} [-E_ini E_INI] [-E_end E_END] [-L] [-V]
                  [-CE] [-W W] [-CT]

Binary decoder for CAEN events from the V1740B digitizer.

options:
  -h, --help    show this help message and exit
  -N N          Number of events to be processed starting from 0.
  -P {0,1}      Plotting option: 0 for single event plot, 1 for multiple event
                plots.
  -E_ini E_INI  Initial event index for multiple event plot option.
  -E_end E_END  Final event index for multiple event plot option.
  -L            Activates logging node in "decoder.log" file.
  -V            Activates verbose mode in console.
  -CE           Call function to check number of events.
  -W W          Set the number of words to check for -CE option.
  -CT           Call function to check TTT of the events.
```