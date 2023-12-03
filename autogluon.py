# Download the dataset, it will be in a .zip file so you'll need to unzip it as well.
kaggle competitions download -c bike-sharing-demand
# If you already downloaded it you can use the -o command to overwrite the file
unzip -o bike-sharing-demand.zip


mkdir -p /root/.kaggle

touch /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

import json

# Using the home directory's path with '~/.kaggle/kaggle.json'
with open("/Users/juanfranceschini/.kaggle/kaggle.json", "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))

import kaggle
kaggle competitions download -c bike-sharing-demand
# If you already downloaded it you can use the -o command to overwrite the file
unzip -o bike-sharing-demand.zip

import pandas as pd
import autogluon

from autogluon.tabular import TabularDataset, TabularPredictor

pip install -U pip
pip install setuptools wheel
pip install -U "mxnet<2.0.0" bokeh==2.0.1

import pandas as pd
from autogluon.tabular import TabularPredictor

# %% 
train = pd.read_csv("train.csv", parse_dates=['datetime'])

train.head()

# Simple output of the train dataset to view some of the min/max/varition of the dataset features.
train.describe()
train.dtypes

# Create the test pandas dataframe in pandas by reading the csv, remember to parse the datetime!
test = pd.read_csv("test.csv", parse_dates=['datetime'])

test.head()

# Same thing as train and test dataset
submission = pd.read_csv("sampleSubmission.csv", parse_dates=['datetime'])
submission.head()

train = train.drop(columns=['casual', 'registered'])
train.head()

train['datetime'] = pd.to_datetime(train['datetime'])

#this predictor achieves 0.67 RMSE
predictor = TabularPredictor(label='count', eval_metric='root_mean_squared_error').fit(
    train_data=train, 
    presets='best_quality', 
    time_limit=600
)


predictions = predictor.predict(test)

predictions.head()
predictions.describe()
# %%

submission["count"] = predictions
submission.to_csv("submission.csv", index=False)


kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "local submission"

kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6


# create a new feature
# Extract hour, day, and month from the datetime column in the train dataset
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['month'] = train['datetime'].dt.month
train['dayofweek'] = train['datetime'].dt.dayofweek
train['year'] = train['datetime'].dt.year
train['week'] = train['datetime'].dt.isocalendar().week

# Extract hour, day, and month from the datetime column in the test dataset
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['month'] = test['datetime'].dt.month
test['dayofweek'] = test['datetime'].dt.dayofweek
test['year'] = test['datetime'].dt.year
test['week'] = test['datetime'].dt.isocalendar().week

test

#transform categorical variables
categorical_features = ['season', 'holiday', 'workingday', 'weather']
for col in categorical_features:
    train[col] = train[col].astype('category')

# Drop the 'datetime', 'casual', and 'registered' columns
train = train.drop(columns=['datetime'])

categorical_features = ['season', 'holiday', 'workingday', 'weather']
for col in categorical_features:
    test[col] = test[col].astype('category')

# Drop the 'datetime', 'casual', and 'registered' columns
test = test.drop(columns=['datetime'])

test.dtypes
train.hist()
train['year'].describe


# Convert datetime to datetime type and extract components
train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['month'] = train['datetime'].dt.month
train['year'] = train['datetime'].dt.year
train['dayofweek'] = train['datetime'].dt.dayofweek

# Convert categorical features
categorical_features = ['season', 'holiday', 'workingday', 'weather']
for col in categorical_features:
    train[col] = train[col].astype('category')

# Drop the 'datetime', 'casual', and 'registered' columns
#train = train.drop(columns=['datetime', 'casual', 'registered'])

# This predictor achieves 0.49 RMSE
predictor = TabularPredictor(label='count', 
                             eval_metric='root_mean_squared_error').fit(
    train_data=train, 
    presets='best_quality',
    time_limit=300)



predictor.fit_summary()
# %%



predictions = predictor.predict(test)
predictions.head()
predictions.describe()
predictions[predictions < 0] = 0
# %%

submission["count"] = predictions
submission.to_csv("submission.csv", index=False)


kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "local submission with new features"

kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
