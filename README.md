# Ceres

Simple script to automate the import of PPMS (DC) data using the corresponding sequence file and raw data. 

### config.ini
Contains all settings and path files used for each script

### import_ppms.py
Responsible for importing raw data, deleting unwanted columns as well as removing outliers above a certain threshold if wanted.
The filtered data will be segmented according to their corresponding tags used in the sequence file and saved in new directory.

### dc_transport.py
Scans the converted data for IV curves, Bscans ....
Calculates ideal maximum excitation current, Hall coefficient, Hall mobility, carrier density and resistivity with their associated standard deviation, if sufficient data is available.
