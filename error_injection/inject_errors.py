from error_generator import Explicit_Missing_Value
from error_generator import Implicit_Missing_Value
from error_generator import White_Noise
from error_generator import Gaussian_Noise
from error_generator import Random_Active_Domain
from error_generator import Similar_Based_Active_Domain
from error_generator import Typo_Keyboard
from error_generator import Typo_Butterfingers
from error_generator import Word2vec_Nearest_Neighbor
from error_generator import Value_Selector
from error_generator import List_selected
from error_generator import Read_Write
from error_generator import Error_Generator

import pandas as pd
import numpy as np



def inject_errors(df_path, methods, error_rates, mute_columns=[]):
    """
    Inject errors on a clean dataset.

    Arguments:
        df_path (str): path to clean dataframe
        methods (list): list of error injection methods
        error_rates (list): error rates corrsponding to each methods
        mute_columns (list): columns that is muted during error injection

    Returns:
        dirty_data (pd.DataFrame): dataframe with injected errors
    """

    dataset, dataframe = Read_Write.read_csv_dataset(df_path)

    for method, error_rate in zip(methods, error_rates):
        myselector = List_selected()
        mygen = Error_Generator()
        new_dataset = mygen.error_generator(method_gen=method, selector=myselector, percentage=error_rate,
                                            dataset=dataset, mute_column=mute_columns)

        dirty_data = Read_Write.write_csv_dataset("./outputs/{}.csv", new_dataset)

        dataset[1:] = dirty_data.values.tolist()
        dataframe = dirty_data

    return dirty_data



if __name__ == '__main__':
    df_path = ''
    methods = [Typo_Keyboard(), Gaussian_Noise(), Implicit_Missing_Value(), Explicit_Missing_Value(), White_Noise()]
    error_rates = [2, 2, 2, 2, 2]

    dirty_df = inject_errors(df_path, methods, error_rates)
