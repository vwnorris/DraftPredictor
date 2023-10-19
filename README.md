# DraftPredictor
## 🚧Construction phase🏗️: 
This is a "just-for-fun" datascience project with a goal of <em>revolutionizing</em> the NFL draft in the coming years. The project contains a dataset of NFL WRs drafted from 2018 to 2020, with their measureables and college statistics. Their wAv/year is then used to train a neural network to predict how valuable new WRs entering the league is going to be. All code is in python, and the data is in xlsx files. 

## Files:
* Data-folder: This folder contains the data used in training and testing the model.
* Regressor: Contains reading the data, plotting PLC, the model, plotting the loss over the epochs as well as training and testing the model. 
* SpreadFiller: Fills the missing values in the data with averages of the other values in the column. 
* SpreadFixer: Converts the measureables of the athletes from ft to cm, and from lbs to kg. 
* NLP: Not used in the project, only for experimenting with adding some NLP features to the model in the future.

## Future features: 
* More information in the dataset:
  * Rating the college
  * Information about position (slot, posession, deep threat, etc)
  * Personality (maturity issues, etc)
  * Injuries
* More draft classes in the dataset (2021, 2022 missing)
* In the future, not only WRs. 

## Running the model:

```py
python3 regressor.py
```

## Contact:
[GitHub](https://github.com/vwnorris)
[LinkedIn]([https://link-url-here.org](https://www.linkedin.com/in/victor-w-t-norris-b58336107/)https://www.linkedin.com/in/victor-w-t-norris-b58336107/)
Email: Vic@Norris.no
