BR_REF_ODS = "BR_Ref_Values/BR_ref_human.ods"

SCENARIOS = {
    "Distance": "1. Distance Scenario",
    "Orientation": "2. Orientation Scenario",
    "Angle": "3. Angle Scenario",
    "Elevated": "4. Elevated"
}

#DISTANCES = ["40 cm", "80 cm", "120 cm", "160 cm"]
RECORDINGS = ["1", "2", "3", "4"]

FS = 20.0
N_FRAMES = 1200
N_CHIRPS = 128
N_RX = 4
N_ADC = 250

OBS_SECONDS = 60
OBS_FRAMES = int(OBS_SECONDS * FS)

W_RR = 1000
W_HR = 400
SW = 20

RR_BAND = (0.1, 0.67)
HR_BAND = (1.0, 3.0)

RR_NFFT = 4096
RR_SNR_THR = 3.0

C = 3e8
FC = 79e9
LAMBDA = C / FC
