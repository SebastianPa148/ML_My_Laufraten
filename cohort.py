import pandas as pd
import config as c

def build_cohort(df: pd.DataFrame):
    """Gemeinsame Kohorte für alle Läufe + Ausschlusstabelle (Flowchart)."""
    needed = [c.GROUP_COL, c.TARGET, c.TIME_COL, c.CO_FEATURE, "Groesse", "BMI"] + c.RATE_COLS
    log = []

    def step(name, d):
        log.append({"Schritt": name, "Messungen": len(d),
                    "Patienten": d[c.GROUP_COL].nunique()})

    step("Rohdaten", df)
    d = df.dropna(subset=needed);                           step("Nach Ausschluss fehlender Werte", d)
    d = d[d[c.TIME_COL] >= c.MIN_TIME_H];                   step(f"Zeit seit Therapiebeginn >= {c.MIN_TIME_H} h", d)
    d = d[(d[c.RATE_COLS] > 0).all(axis=1)];                step("Keine der 4 Laufraten = 0", d)
    d = d[d[c.TARGET] > c.LOQ];                             step(f"Spiegel > Bestimmungsgrenze ({c.LOQ} mg/L)", d)
    d = d[d[c.TARGET] <= c.MAX_SPIEGEL];                    step(f"Spiegel <= {c.MAX_SPIEGEL} mg/L", d)
    d = d[d[c.CO_FEATURE] <= c.MAX_CG_GFR];                 step(f"CG_GFR <= {c.MAX_CG_GFR} ml/min", d)
    d = d[(d[c.RATE_COLS] <= c.MAX_RATE).all(axis=1)];      step(f"Alle 4 Laufraten <= {c.MAX_RATE} ml/h", d)
    d = d[d["Groesse"] >= c.MIN_HEIGHT_CM];                 step(f"Größe >= {c.MIN_HEIGHT_CM} cm", d)
    d = d[d["BMI"] <= c.MAX_BMI];                           step(f"BMI <= {c.MAX_BMI}", d)

    return d.reset_index(drop=True), pd.DataFrame(log)