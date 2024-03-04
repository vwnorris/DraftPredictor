import pandas as pd

# load the excel file
def fixer(filename :str):
    df = pd.read_excel(filename)

    # define a function to convert height from feet and inches to cm
    def convert_to_cm(height):
        if isinstance(height, str) and "'" in height:  # only attempt to split if height is a string and contains an apostrophe
            parts = height.split("'")

            # Handle cases like "6'"
            feet = int(parts[0])  # this part should always be present based on your input data
            inches = 0  # default value

            if len(parts) > 1 and parts[1].strip():  # check if inches part is present and not empty
                inches = int(parts[1].replace('"', '').strip())  # remove double quotes and white spaces if any

            return (feet * 12 + inches) * 2.54
        else:  
            # if height is not a string, or doesn't contain an apostrophe, return it as is
            return height  # or replace with a default value or NaN

    def convert_to_kg(weight):
        if pd.notna(weight) and weight > 140:  # Check if the value is not a NaN
            return weight * 0.453592
        else:
            return weight

    # apply the function to the height column
    df['height'] = df['height'].fillna('0\'0"')  # replace NaNs with '0\'0"'
    df['height'] = df['height'].apply(convert_to_cm)

    # apply the function to the weight column
    df['weight'] = df['weight'].apply(convert_to_kg)

    # write the DataFrame back to an excel file
    df.to_excel(filename, index=False)

    print(df)

fixer("data/rookiesFilled.xlsx")