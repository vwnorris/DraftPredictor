import pandas as pd

# load the excel file
def fixer(filename :str):
    df = pd.read_excel(filename)

    # define a function to convert height from feet and inches to cm
    def convert_to_cm(height):
        if isinstance(height, str) and "'" in height:  # only attempt to split if height is a string and contains an apostrophe
            feet, inches = height.split("'")
            inches = inches.replace('"', '')
            return (int(feet) * 12 + int(inches)) * 2.54
        else:  # if height is not a string, or doesn't contain an apostrophe, return it as is
            return height  # or replace with a default value or NaN

    def convert_to_kg(weight):
        if pd.notna(weight):  # Check if the value is not a NaN
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

fixer("crashCourse/mlModels/rookies.xlsx")