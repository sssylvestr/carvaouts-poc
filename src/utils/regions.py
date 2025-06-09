import pandas as pd

REGION_CODES = ['AUST', 'BELG', 'CZREP','DEN', 'ESTNIA','FIN', 'FRA','GFR', 'ITALY', 'LATV', 'IRE', 'ICEL', 'LITH', 'LUX', 'MALTA', 'NETH', 'NORW', 'POL', 'LARIO', 'SWED', 'SWITZ', 'UK', 'VIEN', 'BRUS', 'PRAGUE',
'COPEN', 'TALLIN','HELSNK', 'PARIS','BERLIN', 'ROME','RIGA', 'DUBLIN','REYK', 'VILNIU', 'LUXCI', 'VALLE', 'AMSTR', 'OSLO', 'WASW', 'MADRD', 'STOCK', 'BERN','DERRY']
REGION_CODES += ["EURZ", "EEURZ", "WEURZ"]
REGION_CODES = [el.lower() for el in REGION_CODES]

def should_keep(region_codes_list, allowed_region_codes=REGION_CODES):
    # Check if any of the region codes in the list are in the exclude list
    return any(code in region_codes_list for code in allowed_region_codes)

def filter_by_region(df: pd.DataFrame) -> pd.DataFrame:
     df['region_codes'] = df['region_codes'].str.strip(',')
     df['region_codes_list'] = df['region_codes'].progress_apply(lambda x: x.split(',') if isinstance(x, str) else [])
     df['regions_relevant'] = df['region_codes_list'].progress_apply(lambda x: should_keep(x))
     
     return df