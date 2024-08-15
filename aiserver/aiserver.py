import threading
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor, XGBClassifier
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient, Point, WritePrecision
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import uniform, randint
import warnings


warnings.filterwarnings('ignore') ## 경고 무시


def makeModel(X,y, scaler): ## 진동 + 온도 위험 판별 모델 생성(센서 별로 하나씩)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)


    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)


    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


    # 데이터 표준화
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # 모델 정의 및 하이퍼파라미터 튜닝
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')


    # 하이퍼파라미터 범위 설정
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'min_child_weight': randint(1, 10)
    }


    # RandomizedSearchCV 실행
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1, verbose=3)
    random_search.fit(X_train_scaled, y_train)


    return random_search.best_estimator_


df = pd.read_csv('overheating_test_data.csv', encoding = 'cp949')


scaler1 = StandardScaler()
X = df[['temp1', 'x1', 'y1', 'z1']]
y = df['sensor1']


model1 = makeModel(X, y,scaler1)
model4 = makeModel(X,y, scaler1)


scaler2 = StandardScaler()
X = df[['temp2', 'x2', 'y2', 'z2']]
y = df['sensor2']


model2 = makeModel(X, y, scaler2)
model5 = makeModel(X,y, scaler2)


scaler3 = StandardScaler()
X = df[['temp3', 'x3', 'y3', 'z3']]
y = df['sensor3']


model3 = makeModel(X, y, scaler3)
model6 = makeModel(X, y, scaler3)


# Configuration
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "TZkhgx4L4Ce-FYOpWpSj7stSm3X1Z11cX1VC1a8OxrQ8zTS09tQVMCHr1QTPsWj4ZQRYAjrB2WtRScuRYGVJBA=="


INFLUX_ORG = "joon"
INFLUX_BUCKET = "modbus2"


# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)


def preprocess_data(data):
    processed_data = []
    for table in data:
        for record in table.records:
            record_time_utc = record.get_time()
            record_time_kst = record_time_utc.astimezone(pytz.timezone('Asia/Seoul'))
            record_time_formatted = record_time_kst.strftime('%Y-%m-%d %H:%M:%S')
            record_field = record.get_field()
            record_value = record.get_value()
            processed_data.append((record_time_formatted, record_field, record_value))
    return processed_data


def vibration_calc(vibration, model, scaler):
    vibration_scaled = scaler.transform(vibration)
    sensor_prediction = model.predict(vibration_scaled)
    return sensor_prediction[0]


def vibration_warning():
    timezone = pytz.timezone('Asia/Seoul')
    try:
        query_api = client.query_api()
        query = '''
            from(bucket: "modbus2")
            |> range(start: -1m)
            |> filter(fn: (r) =>
                r._field == "Temperature1" or r._field == "x1" or r._field == "y1" or r._field == "z1" or
                r._field == "Temperature2" or r._field == "x2" or r._field == "y2" or r._field == "z2" or
                r._field == "Temperature3" or r._field == "x3" or r._field == "y3" or r._field == "z3" or
                r._field == "Temperature4" or r._field == "x4" or r._field == "y4" or r._field == "z4" or
                r._field == "Temperature5" or r._field == "x5" or r._field == "y5" or r._field == "z5" or
                r._field == "Temperature6" or r._field == "x6" or r._field == "y6" or r._field == "z6"
            )
            |> last()
        '''
   
        tables = query_api.query(query, INFLUX_ORG)
   
        data = preprocess_data(tables)
   
        df = pd.DataFrame(data, columns=['time', 'sensor', 'value'])
        df.loc[0, 'value'] = round(df.loc[0, 'value'] / 10)
        vibration_1 = vibration_calc([[df.loc[0, 'value'], df.loc[6, 'value'], df.loc[12, 'value'], df.loc[18, 'value']]], model1, scaler1)    
        vibration_2 = vibration_calc([[df.loc[1, 'value'], df.loc[7, 'value'], df.loc[13, 'value'], df.loc[19, 'value']]], model2, scaler2)    
        vibration_3 = vibration_calc([[df.loc[2, 'value'], df.loc[8, 'value'], df.loc[14, 'value'], df.loc[20, 'value']]], model3, scaler3)    
        vibration_4 = vibration_calc([[df.loc[3, 'value'], df.loc[9, 'value'], df.loc[15, 'value'], df.loc[21, 'value']]], model1, scaler1)    
        vibration_5 = vibration_calc([[df.loc[4, 'value'], df.loc[10, 'value'], df.loc[16, 'value'], df.loc[22, 'value']]], model2, scaler2)    
        vibration_6 = vibration_calc([[df.loc[5, 'value'], df.loc[11, 'value'], df.loc[17, 'value'], df.loc[23, 'value']]], model3, scaler3)    

        print(vibration_1, vibration_2, vibration_3, vibration_4, vibration_5, vibration_6)


        write_api = client.write_api(write_options=SYNCHRONOUS)
       
        data_point = {
            "measurement": "predict",
            "tags": {"tag1": "value1"},
            "time": datetime.now(timezone),
            "fields": {"sensor1": vibration_1, "sensor2": vibration_2, "sensor3": vibration_3, "sensor4": vibration_4, "sensor5": vibration_5, "sensor6": vibration_6
}
        }

        try:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=data_point)
            return True
        except Exception as e:
            print(f"InfluxDB 쓰기 중 오류 발생: {str(e)}")
            return False
   
    except Exception as e:
        print(f"Error during vibration warning: {str(e)}")
        return False



## xgboost regression code
def prediction_temp(prediction_times, data, y):
    predictions = []
    for prediction_time in prediction_times:
        hour = prediction_time.hour
        minute = prediction_time.minute
        second = prediction_time.second
        X_pred = np.array([[hour, minute, second]])
        X = data[['hour', 'minute', 'second']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # NaN 값 처리
        y_train = y_train.fillna(y_train.mean())  # 예시로 평균으로 대체하는 방식

        model = XGBRegressor(n_estimators=100, learning_rate=0.3, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=1)
        model.fit(X_train, y_train)
        tomorrow_temperature = model.predict(X_pred)[0]
        predictions.append((prediction_time, tomorrow_temperature))
    return predictions



def insert_influx(predictions, dataname):
    write_api = client.write_api(write_options=SYNCHRONOUS)
    for time, data in predictions:
        data_point = {
            "measurement": "predict02",
            "tags": {"tag1": "value1"},
            "time": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "fields": {dataname: data}
        }
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=data_point)
    return True


def schedule_and_predict_temperatures():
    while True:
        now = time.localtime()
        if now.tm_hour == 0 and now.tm_min >= 1 and now.tm_min <= 2:
            try:
                print("temp")
                predict_temperatures()
            except Exception as e:
                print(f"Error during temperature prediction: {str(e)}")
            time.sleep(1)  # 다음날 2시 10분까지 대기
        else:
            try:
                vibration_warning()  # 진동 경고 실행
                time.sleep(10)
            except Exception as e:
                print(f"Error during vibration warning: {str(e)}")


def predict_temperatures():
    print("진입")
    utc = pytz.UTC
    kst = pytz.timezone("Asia/Seoul")
    today_kst = datetime.now(kst).replace(hour=0, minute=0, second=0, microsecond=0)
    today_utc = today_kst.astimezone(utc)
    start_time = (today_utc - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = today_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

    query_api = client.query_api()
    query = '''
    from(bucket: "modbus2")
        |> range(start: -30d)
        |> filter(fn: (r) => r._field == "Temperature3" or r._field == "Temperature1" or r._field == "Temperature2" or
                            r._field == "Temperature4" or r._field == "Temperature5" or r._field == "Temperature6" or
                            r._field == "CPU Temperature1" or r._field == "CPU Temperature2" or r._field == "CPU Temperature3" or
                            r._field == "CPU Temperature4" or r._field == "CPU Temperature5" or r._field == "CPU Temperature6")
    '''
    tables = query_api.query(query, INFLUX_ORG)
    data = preprocess_data(tables)
    df = pd.DataFrame(data, columns=['time', 'temperature', 'value'])
    df = df.pivot(index='time', columns='temperature', values='value').reset_index()
    data = df
    data["Temperature1"] = data["Temperature1"] / 10
    data["Temperature2"] = data["Temperature2"] / 10
    data["Temperature3"] = data["Temperature3"] / 10
    data["Temperature4"] = data["Temperature4"] / 10
    data["Temperature5"] = data["Temperature5"] / 10
    data["Temperature6"] = data["Temperature6"] / 10
    y1_temp = data["Temperature1"]
    y2_temp = data["Temperature2"]
    y3_temp = data["Temperature3"]
    y4_temp = data["Temperature4"]
    y5_temp = data["Temperature5"]
    y6_temp = data["Temperature6"]
    y_cpu1_temp = data["CPU Temperature1"]
    y_cpu2_temp = data["CPU Temperature2"]
    y_cpu3_temp = data["CPU Temperature3"]
    y_cpu4_temp = data["CPU Temperature4"]
    y_cpu5_temp = data["CPU Temperature5"]
    y_cpu6_temp = data["CPU Temperature6"]

    data['time'] = pd.to_datetime(data['time'])
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['second'] = data['time'].dt.second
    data['time'] = pd.to_datetime(data['time'])
    start_date = data['time'].max() + pd.Timedelta(seconds=6000)
    end_date = start_date + pd.Timedelta(days=1)
    prediction_times = pd.date_range(start=start_date, end=end_date, freq='6000s')

    try:
        predict_temp1 = prediction_temp(prediction_times, data, y1_temp)
        predict_temp2 = prediction_temp(prediction_times, data, y2_temp)
        predict_temp3 = prediction_temp(prediction_times, data, y3_temp)
        predict_temp4 = prediction_temp(prediction_times, data, y4_temp)
        predict_temp5 = prediction_temp(prediction_times, data, y5_temp)
        predict_temp6 = prediction_temp(prediction_times, data, y6_temp)
        predict_cpu1_temp = prediction_temp(prediction_times, data, y_cpu1_temp)
        predict_cpu2_temp = prediction_temp(prediction_times, data, y_cpu2_temp)
        predict_cpu3_temp = prediction_temp(prediction_times, data, y_cpu3_temp)
        predict_cpu4_temp = prediction_temp(prediction_times, data, y_cpu4_temp)
        predict_cpu5_temp = prediction_temp(prediction_times, data, y_cpu5_temp)
        predict_cpu6_temp = prediction_temp(prediction_times, data, y_cpu6_temp)

        insert_influx(predict_temp1, "predict_temperature1")
        insert_influx(predict_temp2, "predict_temperature2")
        insert_influx(predict_temp3, "predict_temperature3")
        insert_influx(predict_temp4, "predict_temperature4")
        insert_influx(predict_temp5, "predict_temperature5")
        insert_influx(predict_temp6, "predict_temperature6")
        insert_influx(predict_cpu1_temp, "predict_cpu_temperature1")
        insert_influx(predict_cpu2_temp, "predict_cpu_temperature2")
        insert_influx(predict_cpu3_temp, "predict_cpu_temperature3")
        insert_influx(predict_cpu4_temp, "predict_cpu_temperature4")
        insert_influx(predict_cpu5_temp, "predict_cpu_temperature5")
        insert_influx(predict_cpu6_temp, "predict_cpu_temperature6")


    except Exception as e:
        print(f"Error during data processing: {str(e)}")






if __name__ == '__main__':
    schedule_and_predict_temperatures()
