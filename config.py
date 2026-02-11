
BASE_RADAR = "Participant 3/1. Distance Scenario"
BASE_HR_GT = "HR_Ref_Values/Participant 3/1. Distance Scenario"
BR_REF_ODS = "BR_Ref_Values/BR_ref_human.ods"

DISTANCES = ["40 cm", "80 cm", "120 cm", "160 cm"]
RECORDINGS = ["1", "2", "3", "4"]

FS = 20.0
N_FRAMES = 1200
N_CHIRPS = 128
N_RX = 4
N_ADC = 250

OBS_SECONDS = 60
OBS_FRAMES = int(OBS_SECONDS * FS)

W_RR = 800  
W_HR = 400   # 20 s 
SW = 20      # 1 s

RR_BAND = (0.1, 0.3)
HR_BAND = (1.0, 3.0)   

C = 3e8
FC = 79e9
LAMBDA = C / FC
