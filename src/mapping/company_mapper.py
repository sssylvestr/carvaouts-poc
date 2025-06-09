import pandas as pd
import logging, os
from typing import Dict
from tqdm import tqdm
tqdm.pandas()

from src.factiva_api.taxonomy import init_client, get_company_code_mapping
logger = logging.getLogger(__name__)


def extract_company_names(company_codes_list, companies_mapping: Dict[str, str]) -> str:
    company_codes_list = company_codes_list.split(",")

    company_names = []
    for code in company_codes_list:
        if code in companies_mapping.keys():
            company_name = companies_mapping.get(code, code)
            company_names.append(company_name)
        else:
            company_names.append(code)
    return ",".join(company_names)


def map_companies(df,col2process="company_codes", new_col="company_names")-> pd.DataFrame:
     user_key = os.getenv("FACTIVA_SNAPSHOTS_USER_KEY")
     client = init_client(user_key) #type: ignore
     companies_mapping = get_company_code_mapping(client)

     df[new_col] = df[col2process].progress_apply(lambda x: extract_company_names(x, companies_mapping))
     return df