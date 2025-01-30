Fridge temperature ~180 mK to ~190 mK throughout experiment.

------------------------------------------------------------------------------------------------------------------------------------
Each folder contains 4 different channels for measurement:

"A-A-" is nothing (ignore)
"B-V-" is current
"C-V-" is X (reflectometry)
"D-V-" is Y (reflectometry)
"PCA--" is PCA (reflectometry)

--------------------------------------------------------------------------------------------------------------------------------------

The convention for reading RL and LR ticks from the different channels is as follows:

For current channel (B): low-middle-high = RL tick and high-middle-low = LR tick for all values of the SNR (folder ids).
For X channel (C): 	 Same as current channel.
For Y channel (D): 	 Same as current except for last two ids in folder where convention is reversed. I.e. low-middle-high = LR tick 			 and high-middle-low = RL tick.
For PCA channel:   	 Same as current channel except for highest SNR (first id in folder) where convention is reversed. low-middle-high 			 = LR tick and high-middle-low = RL tick.

Here RL tick means hole is transported from drain to source and LR tick means hole is transported from source to drain.

---------------------------------------------------------------------------------------------------------------------------------------


Bias folder names are relative to actual bias of sensor dot (dac 2) = -0.075 mV.

Actual 		Relative
bias [mV]	bias [mV]
==========================
+0.880		+0.955
+0.780		+0.855
+0.630		+0.705
+0.380		+0.455
+0.130		+0.205
-0.050		+0.025
-0.060		+0.015
-0.070		+0.005
-0.075		 0