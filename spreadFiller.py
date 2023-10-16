print("Hello world")

import pandas as pd
import numpy as np

def filler(filename: str):
    # Load the spreadsheet
    xlsx = pd.ExcelFile(filename)

    # Load a sheet into a DataFrame by name
    df = pd.read_excel(xlsx, 'Sheet1')  # replace 'Sheet1' with your actual sheet name

    # Fill the NaN values with the column mean
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col].fillna(df[col].mean(), inplace=True)

    
    newFile = filename.replace('.xlsx', 'Filled.xlsx')
    # Save the result back to Excel
    df.to_excel(newFile, index=False)

filler('crashCourse/mlModels/rookies.xlsx')