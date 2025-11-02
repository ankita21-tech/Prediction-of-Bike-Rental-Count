import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle

warnings.filterwarnings('ignore')


def train_and_save(df, model_path, scaler_path):
    """Enhanced training with feature engineering"""
    # Converting date to Pandas datetime format
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])
    
    # Map categorical columns to strings
    if 'season' in df.columns:
        df.season = df.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})
    if 'weathersit' in df.columns:
        df.weathersit = df.weathersit.map({1:'Best', 2:'Neutral', 3:'Bad', 4:'Worse'})
    if 'mnth' in df.columns:
        df.mnth = df.mnth.map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
    if 'weekday' in df.columns:
        df.weekday = df.weekday.map({1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',0:'Sun'})
    
    # Feature engineering: Add interaction features
    if 'temp' in df.columns and 'hum' in df.columns:
        df['temp_hum'] = df['temp'] * df['hum']
    if 'temp' in df.columns and 'windspeed' in df.columns:
        df['temp_wind'] = df['temp'] * df['windspeed']
    if 'yr' in df.columns and 'mnth' in df.columns:
        df['yr_mnth_interaction'] = df['yr'].astype(str) + '_' + df['mnth'].astype(str)
    
    # Drop unnecessary columns
    for col in ['instant','dteday','casual','registered','atemp']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Create dummies for categorical features
    def dummies(x, dataframe):
        if x in dataframe.columns:
            temp = pd.get_dummies(dataframe[x], drop_first=True)
            dataframe = pd.concat([dataframe, temp], axis=1)
            dataframe.drop([x], axis=1, inplace=True)
        return dataframe
    
    for col in ['season', 'mnth', 'weekday', 'weathersit', 'yr_mnth_interaction']:
        df = dummies(col, df)
    
    # Train-test split
    df_train, df_test = train_test_split(df, train_size=0.7, random_state=100)
    
    # Scale numeric features
    scaler = MinMaxScaler()
    need_rescale = [c for c in ['temp', 'hum', 'windspeed', 'temp_hum', 'temp_wind'] if c in df_train.columns]
    if need_rescale:
        df_train[need_rescale] = scaler.fit_transform(df_train[need_rescale])
    
    y_train = df_train.pop('cnt')
    X_train = df_train
    
    # Train model with better hyperparameters
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=100
    )
    gb.fit(X_train, y_train)
    
    # Save model and scaler
    pickle.dump(gb, open(model_path, 'wb'))
    pickle.dump(scaler, open(scaler_path, 'wb'))
    
    print(f"✓ Model saved to {model_path} with {len(X_train.columns)} features")
    return X_train.columns.tolist()


if __name__ == "__main__":
    print("Training Day model...")
    df_day = pd.read_csv('day.csv')
    day_features = train_and_save(df_day, 'model_day.pkl', 'scaler_day.pkl')
    
    print("\nTraining Hour model...")
    df_hour = pd.read_csv('hour.csv')
    hour_features = train_and_save(df_hour, 'model_hour.pkl', 'scaler_hour.pkl')
    
    print(f"\n✓ Training complete!")
    print(f"Day model features: {len(day_features)}")
    print(f"Hour model features: {len(hour_features)}")
