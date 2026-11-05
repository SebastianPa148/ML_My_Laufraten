from pathlib import Path

SEED = 42
DATA_FILE = Path("/Users/sebastian/ML_My/data/Mero_2_my_komplett_red_long1.csv")  # anpassen

GROUP_COL   = "encounterID"
TARGET      = "AB_Spiegel"
TIME_COL    = "Time_Start_AB_M"
CO_FEATURE  = "CG_GFR"

RATE_COLS = ["Laufrate_6Uhr", "Laufrate_Av_6h",
             "Laufrate_Av_12h", "Laufrate_Av_24h"]

# Feature-Sets: Name -> Spalten. Erweiterbar für spätere Features.
FEATURE_SETS = {"Referenz_nur_CG_GFR": [CO_FEATURE]}
FEATURE_SETS.update({rate: [rate, CO_FEATURE] for rate in RATE_COLS})

MIN_TIME_H   = 24      # Time_Start_AB_M muss größer sein
OUTER_FOLDS  = 5
INNER_FOLDS  = 5
TARGET_RANGES = [(8, 40), (12, 40)]   # für später

# Plausibilitätsgrenzen (klinisch begründet, vor der Modellierung festgelegt)
LOQ            = 1.1    # Bestimmungsgrenze Assay (mg/L); Werte <= LOQ werden ausgeschlossen
MAX_SPIEGEL    = 200    # mg/L, darüber Mess-/Abnahmefehler
MAX_CG_GFR     = 400    # ml/min
MAX_RATE       = 30     # ml/h, gilt für alle vier Laufraten
MIN_HEIGHT_CM  = 140    # Sicherheitscheck nach Korrektur der Originaldatei
MAX_BMI        = 90     # Sicherheitscheck nach Korrektur der Originaldatei

DATA_FILE   = Path("/Users/sebastian/ML_My/data/Mero_2_my_komplett_red_long1.csv")  # korrigierte Version!
FOLDS_FILE  = "results/kohorte_mit_folds.csv"

FOLDS_FILE = "results/kohorte_mit_folds.csv"
N_STRATA = 5   # Quantil-Bins des Spiegels für die Stratifizierung