# Seismic Data Analysis with Streaming Machine Learning

This project focuses on analyzing seismic velocity data from the Moon and Mars. It utilizes the Adaptive Random Forest algorithm for real-time earthquake detection using streaming machine learning techniques. The project is implemented in Python and is structured as a Jupyter Notebook for interactive, step-by-step execution.

## Ensure to download NASA data (https://wufs.wustl.edu/SpaceApps/data/space_apps_2024_seismic_detection.zip) and put the path inside this repo.

## Prerequisites

Before running the notebook, ensure to follow these steps:

```bash
conda create -n *name_your_environment*
python 3.10.12
conda activate *name_your_environment*
pip install numpy==1.26.4
pip install scikit-multiflow
```

Install the necessary Python libraries using:
```bash
pip install -r requirements.txt
```

requirements.txt:
```bash
bash
numpy
pandas
matplotlib
scikit-learn
scipy
obspy
scikit-multiflow
```


## Data Preparation
### Lunar Data
The lunar data consists of CSV files inside NASA. Each CSV file contains seismic velocity data with the following columns:

- time_abs(%Y-%m-%dT%H:%M:%S.%f)
- time_rel(sec)
- velocity(m/s)
- filename
- Steps to Prepare Lunar Data:

Concatenate CSV Files: The script walks through the data/lunar/test/ directory, reads each CSV file, and concatenates them into a single DataFrame.

Save Concatenated Data: The combined DataFrame is saved as data/lunar/test/data/lunar_catalogs.csv.
main_path = 'data/lunar/test/'
```bash
list_dfs = []

for root, dirs, files in os.walk(main_path):
    for file in files:
        if file.endswith('.csv'):
            trackfile = os.path.join(root, file)
            df = pd.read_csv(trackfile)
            df['filename'] = file
            list_dfs.append(df)

df_concat = pd.concat(list_dfs, ignore_index=True)
df_concat.to_csv('data/lunar/test/data/lunar_catalogs.csv', index=False)
```

## Martian Data
Similarly, Martian seismic data is located in the data/mars/test/data/ directory.

Steps to Prepare Martian Data:

Concatenate CSV Files: The script processes each CSV file in the Martian data directory and combines them.
Save Concatenated Data: The merged DataFrame is saved as data/mars/test/data/mars.csv.

## Usage
Prepare the Data: Ensure that the lunar and Martian seismic data CSV files are placed in their respective directories as specified.

Run the Notebook: Open the Jupyter Notebook and execute the cells step by step to process the data and run the machine learning model.

Monitor Output: The evaluation metrics and model performance will be displayed in the notebook output.

## Conclusion:
The model achieves high accuracy due to the imbalance in the dataset (few seismic events compared to non-events).

Precision and recall are low, indicating difficulty in correctly identifying seismic events.

Further data balancing and feature engineering may improve model performance.
