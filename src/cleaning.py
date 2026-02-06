import pandas as pd

COLUMN_MAP = {
    'Tidstämpel': 'date',
    'Om INTE idag, ange datum:': 'retro_date',
    'Tid (välj närmaste halvtimme)': 'time',
    'Aktivitet (engelska sist)': 'activity',
    'Antal deltagare': 'participants',
    'Egen bedömning av aktiviteten': 'grade',
    'Följande anpassningar behövdes (beskriv gärna orsak):': 'adjustments',
    'Din bedömning av deltagarnas intressenivå': 'interest_level',
    'Tips eller något annat du vill dela med dig av ?': 'tips',
    'Namn (frivilligt)': 'name'
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df.rename(columns=COLUMN_MAP)