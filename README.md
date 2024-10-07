# Seismic Data Analysis with Streaming Machine Learning

This project focuses on analyzing seismic velocity data from the Moon and Mars. It utilizes the Adaptive Random Forest algorithm for real-time earthquake detection using streaming machine learning techniques. The project is implemented in Python and is structured as a Jupyter Notebook for interactive, step-by-step execution.

## Prerequisites

Before running the notebook, ensure you have the following installed:

- Python 3.10.12
- Jupyter Notebook

Install the necessary Python libraries using:

```bash
pip install -r requirements.txt
```
requirements.txt:
```bash
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
The lunar data consists of CSV files located in the data/lunar/test/ directory. Each CSV file contains seismic velocity data with the following columns:

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

## Data Processing
### Scaling and Smoothing
Scaling: The velocity data is scaled using MinMaxScaler to normalize the values between 0 and 1.

Smoothing: A rolling mean is applied to the scaled data to smooth out short-term fluctuations using a window size of 10.

```bash
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data['velocity(m/s)'] = scaler.fit_transform(data[['velocity(m/s)']])
window_size = 10
data['velocity_smooth'] = data['velocity(m/s)'].rolling(window=window_size).mean()
```
## Peak Detection
Finding Peaks: The find_peaks function from scipy.signal is used to identify peaks in the smoothed velocity data, which may indicate seismic events.

```bash
from scipy.signal import find_peaks

peaks, _ = find_peaks(data['velocity_smooth'], height=0.01)
data['is_sismo'] = 0
data.loc[peaks, 'is_sismo'] = 1
```
## Z-Score Calculation
Calculating Z-Score: The Z-score helps in identifying outliers in the data, which can correspond to seismic events.
```bash
from scipy import stats

data['z_score'] = np.abs(stats.zscore(data['velocity(m/s)']))
sismos = data[data['z_score'] > 0.1]
```
## Machine Learning Model
### Adaptive Random Forest Classifier

The project employs the Adaptive Random Forest Classifier from the scikit-multiflow library to handle streaming data and concept drift, which is crucial for real-time seismic event detection.
```bash
from skmultiflow.data import DataStream
from skmultiflow.meta import AdaptiveRandomForestClassifier

X = sismos[['time_rel(sec)', 'velocity(m/s)']].values
y = sismos['is_sismo'].values

stream = DataStream(X, y)
model = AdaptiveRandomForestClassifier()
```
## Evaluation
Evaluator: The EvaluatePrequential method is used for prequential (interleaved test-then-train) evaluation, suitable for streaming data.
Metrics: Accuracy, Precision, Recall, F1-score
```bash
from skmultiflow.evaluation import EvaluatePrequential

evaluator = EvaluatePrequential(
    pretrain_size=100,
    max_time=1000,
    max_samples=len(X),
    batch_size=1,
    n_wait=100,
    show_plot=False,
    metrics=['accuracy', 'precision', 'recall', 'f1']
)

evaluator.evaluate(stream=stream, model=[model], model_names=['ARF'])
```
## Usage
Prepare the Data: Ensure that the lunar and Martian seismic data CSV files are placed in their respective directories as specified.

Run the Notebook: Open the Jupyter Notebook and execute the cells step by step to process the data and run the machine learning model.

Monitor Output: The evaluation metrics and model performance will be displayed in the notebook output.

## Conclusion:
The model achieves high accuracy due to the imbalance in the dataset (few seismic events compared to non-events).

Precision and recall are low, indicating difficulty in correctly identifying seismic events.

Further data balancing and feature engineering may improve model performance.


