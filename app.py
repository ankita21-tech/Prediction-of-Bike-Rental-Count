import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask import redirect, url_for, session, flash
import pickle


app= Flask(__name__, static_url_path='/static')
# Use a rotating secret key in dev so old cookies don't auto-login across restarts
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(24)
# Load both models and scalers
with open('model_day.pkl', 'rb') as f:
    model_day = pickle.load(f)
with open('scaler_day.pkl', 'rb') as f:
    scaler_day = pickle.load(f)
with open('model_hour.pkl', 'rb') as f:
    model_hour = pickle.load(f)
with open('scaler_hour.pkl', 'rb') as f:
    scaler_hour = pickle.load(f)

# Preload datasets for insights and stats
FEATURES_ORDER = ['Best', 'Neutral', 'spring', 'temp', 'winter', 'summer', 'hum', 'Jul', 'Sep', 'windspeed', 'yr', 'holiday']

def _prep_df_for_features(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    # mapping similar to training
    if 'season' in dfc.columns:
        dfc.season = dfc.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})
    if 'weathersit' in dfc.columns:
        dfc.weathersit = dfc.weathersit.map({1:'Best', 2:'Neutral', 3:'Bad', 4:'Worse'})
    if 'mnth' in dfc.columns:
        dfc.mnth = dfc.mnth.map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
    if 'weekday' in dfc.columns:
        dfc.weekday = dfc.weekday.map({1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',0:'Sun'})
    for col in ['instant','dteday','casual','registered','atemp']:
        if col in dfc.columns:
            dfc = dfc.drop(col, axis=1)
    def dummies(x,dataframe):
        if x in dataframe.columns:
            temp = pd.get_dummies(dataframe[x], drop_first=True)
            dataframe = pd.concat([dataframe, temp], axis=1)
            dataframe.drop([x], axis=1, inplace=True)
        return dataframe
    for col in ['season','mnth','weekday','weathersit']:
        dfc = dummies(col, dfc)
    return dfc

try:
    df_day_info = pd.read_csv('day.csv')
    df_hour_info = pd.read_csv('hour.csv')
    df_day_feat_df = _prep_df_for_features(df_day_info)
    df_hour_feat_df = _prep_df_for_features(df_hour_info)
    feature_names_day = [c for c in FEATURES_ORDER if c in df_day_feat_df.columns]
    feature_names_hour = [c for c in FEATURES_ORDER if c in df_hour_feat_df.columns]
except Exception:
    df_day_info = pd.DataFrame()
    df_hour_info = pd.DataFrame()
    feature_names_day = FEATURES_ORDER
    feature_names_hour = FEATURES_ORDER

@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/index')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    # allow pre-selecting dataset via query param (e.g., /index?dataset=hour)
    selected = request.args.get('dataset', 'day')
    return render_template('index.html', selected_dataset=selected)

@app.route('/insights')
def insights():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Feature importances
    def names_values(model, names, k=10):
        try:
            imps = list(model.feature_importances_)
            n = min(len(imps), len(names))
            pairs = list(zip(names[:n], imps[:n]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:k]
            labels = [p[0] for p in pairs]
            values = [round(float(p[1]) * 100, 2) for p in pairs]  # Convert to percentage
            return labels, values
        except Exception:
            return [], []
    
    day_labels, day_values = names_values(model_day, feature_names_day, k=10)
    hour_labels, hour_values = names_values(model_hour, feature_names_hour, k=10)
    
    # Data statistics from datasets
    stats = {}
    try:
        # Rental patterns by season
        season_counts = df_hour_info.groupby('season')['cnt'].mean().round(0).tolist()
        season_labels = ['Spring', 'Summer', 'Fall', 'Winter']
        
        # Rental patterns by weather
        weather_counts = df_hour_info.groupby('weathersit')['cnt'].mean().round(0).tolist()
        weather_labels = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain'][:len(weather_counts)]
        
        # Hourly patterns (for hour dataset)
        if 'hr' in df_hour_info.columns:
            hourly_avg = df_hour_info.groupby('hr')['cnt'].mean().round(0).tolist()
            hour_labels_chart = list(range(24))
        else:
            hourly_avg = []
            hour_labels_chart = []
        
        # Working day vs Holiday
        workday_avg = df_hour_info[df_hour_info['workingday'] == 1]['cnt'].mean()
        holiday_avg = df_hour_info[df_hour_info['workingday'] == 0]['cnt'].mean()
        
        # Temperature correlation bins
        df_hour_info['temp_bin'] = pd.cut(df_hour_info['temp'], bins=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'])
        temp_rentals = df_hour_info.groupby('temp_bin', observed=True)['cnt'].mean().round(0).tolist()
        temp_labels = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'][:len(temp_rentals)]
        
        stats = {
            'season_counts': season_counts,
            'season_labels': season_labels,
            'weather_counts': weather_counts,
            'weather_labels': weather_labels,
            'hourly_avg': hourly_avg,
            'hour_labels_chart': hour_labels_chart,
            'workday_avg': round(workday_avg, 0),
            'holiday_avg': round(holiday_avg, 0),
            'temp_rentals': temp_rentals,
            'temp_labels': temp_labels,
            'total_rentals': int(df_hour_info['cnt'].sum()),
            'avg_daily_rentals': int(df_day_info['cnt'].mean()),
            'peak_hour': int(df_hour_info.groupby('hr')['cnt'].mean().idxmax()) if 'hr' in df_hour_info.columns else 17,
            'best_season': season_labels[season_counts.index(max(season_counts))],
        }
    except Exception as e:
        print(f"Error computing stats: {e}")
        stats = {}
    
    return render_template('insights.html', 
                           day_labels=day_labels, day_values=day_values,
                           hour_labels=hour_labels, hour_values=hour_values,
                           stats=stats)

@app.route('/dataset')
def dataset_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    def stats(df):
        if df is None or df.empty:
            return {"status": "unavailable"}
        out = {"rows": len(df), "columns": len(df.columns)}
        for col in ['temp','hum','windspeed','cnt']:
            if col in df.columns:
                out[f"avg {col}"] = round(float(df[col].mean()), 3)
        return out
    day_rows = len(df_day_info) if not df_day_info.empty else 0
    day_cols = len(df_day_info.columns) if not df_day_info.empty else 0
    hour_rows = len(df_hour_info) if not df_hour_info.empty else 0
    hour_cols = len(df_hour_info.columns) if not df_hour_info.empty else 0
    return render_template('dataset.html',
                           day_rows=day_rows,
                           day_cols=day_cols,
                           hour_rows=hour_rows,
                           hour_cols=hour_cols,
                           day_stats=stats(df_day_info),
                           hour_stats=stats(df_hour_info))

@app.route('/contact', methods=['GET','POST'])
def contact():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    msg = ''
    if request.method == 'POST':
        # In a real app, you'd send an email or store the message
        msg = 'Thanks for reaching out! We\'ll get back to you soon.'
    return render_template('contact.html', msg=msg)

@app.route('/chatbot')
def chatbot():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/chatbot_api', methods=['POST'])
def chatbot_api():
    if not session.get('logged_in'):
        return jsonify({"error": "unauthorized"}), 401
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get('message') or '').lower()
        if any(k in text for k in ['hello','hi','hey']):
            reply = 'Hi! You can ask me about predictions, datasets, or model features.'
        elif 'predict' in text:
            reply = 'Use the Prediction tab. Choose Day or Hour dataset, fill inputs, and hit Predict.'
        elif 'dataset' in text or 'data set' in text:
            reply = f"Day rows: {len(df_day_info) if not df_day_info.empty else 0}, Hour rows: {len(df_hour_info) if not df_hour_info.empty else 0}."
        elif 'feature' in text or 'important' in text:
            td = ', '.join([n for n,_ in sorted(zip(feature_names_day, getattr(model_day, 'feature_importances_', [])), key=lambda x: x[1], reverse=True)[:3]]) if hasattr(model_day,'feature_importances_') else 'temp, hum, windspeed'
            th = ', '.join([n for n,_ in sorted(zip(feature_names_hour, getattr(model_hour, 'feature_importances_', [])), key=lambda x: x[1], reverse=True)[:3]]) if hasattr(model_hour,'feature_importances_') else 'temp, hum, windspeed'
            reply = f"Top features (day): {td}. Top features (hour): {th}."
        else:
            reply = "I'm a demo assistant. Try asking: 'What are top features?' or 'How to predict?'"
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"})

@app.route('/shuffle')
def predictionpage():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('shuffle.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Determine which dataset/model to use
    dataset = request.args.get('dataset', request.form.get('dataset', 'day'))
    if dataset not in ['day', 'hour']:
        dataset = 'day'
    if dataset == 'day':
        model = model_day
        scaler = scaler_day
        expected = ['weather','Seasons','temp','hum','Month','windspeed','yr','holiday']
        rescaling_cols=['temp', 'hum', 'windspeed']
    else:
        model = model_hour
        scaler = scaler_hour
        expected = ['weather','Seasons','temp','hum','Month','windspeed','yr','holiday','hr','weekday','workingday']
        rescaling_cols=['temp', 'hum', 'windspeed']

    # Build input dict with proper types
    input_data = {}
    for k in expected:
        v = request.form.get(k, None)
        if v is None:
            input_data[k] = 0.0
        else:
            try:
                input_data[k] = float(v)
            except Exception:
                input_data[k] = 0.0

    # Create DataFrame with raw input
    df = pd.DataFrame([input_data])
    
    # Map numeric values to categorical strings (like training does)
    season_map = {1:'spring', 2:'summer', 3:'fall', 4:'winter'}
    weather_map = {1:'Best', 2:'Neutral', 3:'Bad', 4:'Worse'}
    month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    
    if 'Seasons' in df.columns:
        df['season'] = df['Seasons'].map(season_map)
        df = df.drop('Seasons', axis=1)
    if 'weather' in df.columns:
        df['weathersit'] = df['weather'].map(weather_map)
        df = df.drop('weather', axis=1)
    if 'Month' in df.columns:
        df['mnth'] = df['Month'].map(month_map)
        df = df.drop('Month', axis=1)
    
    # Create interaction features BEFORE one-hot encoding
    if 'temp' in df.columns and 'hum' in df.columns:
        df['temp_hum'] = df['temp'] * df['hum']
    if 'temp' in df.columns and 'windspeed' in df.columns:
        df['temp_wind'] = df['temp'] * df['windspeed']
    if 'yr' in df.columns and 'mnth' in df.columns:
        df['yr_mnth_interaction'] = df['yr'].astype(int).astype(str) + '_' + df['mnth'].astype(str)
    
    # Create dummy variables using the same method as training
    # Note: Only create dummies for categorical columns, not mnth (since we use yr_mnth_interaction instead)
    def dummies(x, dataframe):
        if x in dataframe.columns:
            temp = pd.get_dummies(dataframe[x], drop_first=True)
            dataframe = pd.concat([dataframe, temp], axis=1)
            dataframe = dataframe.drop(x, axis=1)
        return dataframe
    
    # Drop mnth before creating dummies since we use yr_mnth_interaction
    if 'mnth' in df.columns:
        df = df.drop('mnth', axis=1)
    
    for col in ['season', 'weathersit', 'yr_mnth_interaction']:
        df = dummies(col, df)
    
    # Debug: print what we have vs what model expects
    print(f"DataFrame columns after dummies: {sorted(df.columns.tolist())}")
    print(f"Model expects: {sorted(model.feature_names_in_.tolist())}")
    
    # Ensure all expected features are present (add missing with 0)
    for feat in model.feature_names_in_:
        if feat not in df.columns:
            df[feat] = 0
    
    # Keep only the features the model expects, in the correct order
    df = df[model.feature_names_in_]
    
    # Scale numeric features
    rescaling_cols_full = ['temp', 'hum', 'windspeed', 'temp_hum', 'temp_wind']
    cols_to_scale = [c for c in rescaling_cols_full if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    #Prediction of the trained model
    prediction= model.predict(df)
    #Output derived from the ML model
    output= round(prediction[0], 2)

    #Output sent to the html page, keep dataset selection
    return render_template('index.html', prediction_text='Prediction: \n {} bike rents.'.format(output), selected_dataset=dataset)


@app.route('/filter', methods=['POST'])
def filter_data():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    dataset = request.form.get('dataset', 'day')
    return redirect(url_for('index', dataset=dataset))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    msg = ''
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        old = request.form.get('old_password')
        new = request.form.get('new_password')
        confirm = request.form.get('confirm_password')
        # Only allow change if old is correct and new matches confirm
        if old == 'admin123' and new and new == confirm:
            msg = 'Password changed (demo: not persisted, resets on restart)'
        else:
            msg = 'Password change failed. Check your entries.'
    return render_template('change_password.html', msg=msg)

if __name__=="__main__":
    app.run(debug=True)