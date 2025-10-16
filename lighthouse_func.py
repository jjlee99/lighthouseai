# lighthouse_func.ipynb
# 라이브러리 설치시 사이킷런 1.3.2버전 먼저 설치
import base64
import os
import io
import sys
import pickle
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
matplotlib.use('Agg') 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from autoviz.AutoViz_Class import AutoViz_Class
from autoviz import data_cleaning_suggestions
from sklearn.metrics import (
            accuracy_score, 
            f1_score, recall_score, 
            precision_score, 
            roc_auc_score, cohen_kappa_score,
            matthews_corrcoef,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            mean_absolute_percentage_error
)


from fairlearn.reductions import ExponentiatedGradient, DemographicParity,EqualizedOdds, BoundedGroupLoss
from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.preprocessing import StandardScaler
import warnings
import pygwalker as pyg
import importlib
from datetime import date
from fairlearn.metrics import *
from contextlib import redirect_stdout
import shap

warnings.filterwarnings(action="ignore")

# 모델 학습에 pycaret기능을 사용하였는데, 이례적으로 pycaret 라이브러리는 분석 문제에 따라 임포트를 달리해줘야만 합니다.
def lighthouse_setup(project_type):
    """
    사용자로부터 문제 유형을 입력받아 적절한 PyCaret 모듈을 동적으로 임포트합니다.
    """
    # project_type = input("문제 유형을 입력하세요 (분류 또는 회귀): ").lower()

    if project_type == "분류":
        module_name = "classification"
    elif project_type == "회귀":
        module_name = "regression"
    else:
        raise ValueError("잘못된 문제 유형입니다. '분류' 또는 '회귀'를 입력하세요.")

    pycaret_module = importlib.import_module(f"pycaret.{module_name}")
    globals().update(pycaret_module.__dict__)
    os.makedirs('html', exist_ok=True)
    os.makedirs('etc', exist_ok=True)
    os.makedirs('json', exist_ok=True)
    os.makedirs('pkl', exist_ok=True)
    os.makedirs('prepro_data', exist_ok=True)
    os.makedirs('ori_data', exist_ok=True)

    print(f"pycaret.{module_name} 모듈이 성공적으로 import 되었습니다.")


# json db저장 코드
def json_api(pid,table_name,json_data):
    json_url = 'http://192.168.10.127:8000/postdb/light/insertFile/json/'
    json_postVal = {
            'val1':pid, # 프로젝트 ID
            'val2':table_name, # json code?
            'val3':json_data # json 데이터 or 텍스트로 변환된 이미지
    }
    response = requests.post(json_url,json = json_postVal)
    if response.status_code == 200:
        data = response.json()  # JSON 형식의 응답 데이터를 파이썬 딕셔너리로 변환
        print(f"Json id {data} 로 저장 완료됨")
    else:
        print(f'오류: {response.status_code}')


# pycaret으로 생성한 성능지표 이미지 db 저장 코드
def pycaret_image_upload(pid, project_type,directory,dataset):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    image_url = 'http://192.168.10.127:8000/postdb/light/insertFile/image/'
    
    # 테이블 이름과 해당하는 이미지 파일 이름을 매핑
    if project_type == '분류':
        table_image_map = {
            "AUC": f"{dataset}_AUC.png",
            "precision Recall": f"{dataset}_Precision Recall.png",
            "Confusion Matrix": f"{dataset}_Confusion Matrix.png",
            "Threshold": f"{dataset}_Threshold.png",
            "Feature_Importance" : f"{dataset}_Feature Importance.png",
            "SHAP summary" :  f"{dataset}_SHAP summary.png"
        }
    elif project_type == '회귀':
        table_image_map = {
            "Residuals" :  f"{dataset}_Residuals.png",
            "Prediction Error" : f"{dataset}_Prediction Error.png",
            "Cooks Distance" : f"{dataset}_Cooks Distance.png",
            "Learning Curve" :  f"{dataset}_Learning Curve.png",
            "Feature Importance." : f"{dataset}_Feature Importance.png"
            }
    else :
        print('분석문제를 다시 확인바랍니다')
    
    for table_name, image_file in table_image_map.items():
        img_path = os.path.join(directory, image_file)
        
        if not os.path.exists(img_path):
            print(f"경고: {img_path} 경로가 존재하지 않습니다.")
            continue
        
        # 이미지 파일 읽기 및 인코딩
        try:
            with open(img_path, 'rb') as img_file:
                image_bytes = img_file.read()
                encoded_string = base64.b64encode(image_bytes).decode('utf-8')
            
            print(f"{image_file}의 인코딩 텍스트:")
            print(encoded_string[:20] + "...")  # 인코딩된 문자열의 일부만 출력
            print()  # 각 이미지 출력 사이에 빈 줄 추가
            
            # API 요청 데이터 준비
            image_postVal = {
                'val1': pid,  # 프로젝트 ID
                'val2': table_name,  # 테이블명
                'val3': encoded_string  # 텍스트로 변환된 이미지
            }
            
            # API 요청 보내기
            response = requests.post(image_url, json=image_postVal)
            if response.status_code == 200:
                data = response.json()  # JSON 형식의 응답 데이터를 파이썬 딕셔너리로 변환
                print(f"{table_name} 저장 완료했습니다: {data}")
                print("\n","-"*100)
            else:
                print(f'{table_name} 오류가 발생했습니다: {response.status_code}')
        
        except FileNotFoundError:
            print(f"오류:  {img_path} 파일이 존재하지 않습니다")
        except requests.exceptions.RequestException as e:
            print(f"{table_name}으로의 api 요청이 실패했습니다: {e}")
        except Exception as e:
            print(f"{image_file}을 저장하는데 예기치못한 오류가 발생했습니다.: {e}")

# pycaret으로 추출한 성능지표를 json으로 변환하는 코드
def convert_pycaretfair_to_json(project_type,pycaret_df, dataset_name):
    # Convert DataFrame to list of dictionaries
    json_data = []
    if project_type == '분류':
        for _, row in pycaret_df.iterrows():
            json_data.append({
               "Model" : row["Model"],
                "Accuracy" : row["Accuracy"],
                "AUC" : row["AUC"],
                "Recall" : row["Recall"],
                "Prec.":row["Prec."] ,
                "F1" : row["F1"],
                "Kappa" :row[ "Kappa"] ,
                "MCC" :row["MCC"] 
            })
        # Convert to JSON string
        json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        # Write to file
        with open(f'json/{dataset_name}_fairness.json', 'w') as file:
            file.write(json_string)
        
        print(f"파일명 json/{dataset_name}_fairness.json으로 json 저장 완료")
        # Optionally, return the JSON string
        return json_string
    elif project_type == '회귀':
        for _, row in pycaret_df.iterrows():
            json_data.append({
               "Model" : row["Model"],
                "MAE" : row["MAE"],
                "MSE" : row["MSE"],
                "RMSE" : row["RMSE"],
                "R2" :row["R2"] ,
                "RMSLE" : row["RMSLE"],
                "MAPE" :row[ "MAPE"]
            })
        # Convert to JSON string
        json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        # Write to file
        with open(f'json/{dataset_name}_fairness.json', 'w') as file:
            file.write(json_string)
        
        print(f"파일명 json/{dataset_name}_fairness.json으로 json 저장 완료")
        # Optionally, return the JSON string
        return json_string

# fairlearn으로 생성한 공정성 데이터프레임을 json으로 변환하는 코드
def convert_fairlearn_to_json(fairlearn_df, dataset_name):
    # Convert DataFrame to list of dictionaries
    json_data = []
    for _, row in fairlearn_df.iterrows():
        json_data.append({
            "date": row['date'],
            "col_nm": row['col_nm'],
            "demographic_parity_difference": row['demographic_parity_difference'],
            "demographic_parity_ratio": row['demographic_parity_ratio'],
            "equalized_odds_difference": row['equalized_odds_difference'],
            "equalized_odds_ratio": row['equalized_odds_ratio']
        })
    
    # Convert to JSON string
    json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
    
    # Write to file
    with open(f'json/{dataset_name}_fairlearn.json', 'w') as file:
        file.write(json_string)
    
    print(f"파일명 json/{dataset_name}_fairlearn.json으로 json 저장 완료")
    
    # Optionally, return the JSON string
    return json_string

# 데이터셋을 불러오는 코드
def load_project(dataset,verboose):
    """
    주어진 이름에 따라 데이터를 로드합니다.
    """
    file_path = f'~/ori_data/{dataset}.csv'
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"데이터셋 '{dataset}'이 성공적으로 로드되었습니다.")
    if verboose == True:
        return df
    else:
        None

# 원천 데이터셋에 대한 시각화.
def lh_ori_vis(dataset,prediction):
    df = pd.read_csv(f'./ori_data/{dataset}.csv', encoding='utf-8')
    print("\n원천 데이터 시각화 1")
    walke1 = pyg.walk(df)
    with open(f"html/{dataset}_before.html", "w", encoding="utf-8") as f:
        f.write(pyg.to_html(df))
    print("-"*100)
    print("\n원천 데이터 시각화 2")
    # bias_before
    AV = AutoViz_Class()
    # 시각화 결과 저장 경로
    save_plot_dir = f"image/before/"
    # 'html' 디렉토리가 없으면 생성
    os.makedirs('image/before', exist_ok=True)
    # 자동 시각화 실행
    # max_rows_analyzed, max_cols_analyzed를 최대로 보여주기위해 150000과 수치를 입력.
    dft = AV.AutoViz(
        filename="",
        sep=",",
        depVar=f"{prediction}",
        dfte=df,
        header=0,
        verbose=2,
        lowess=False,
        chart_format="png",
        max_rows_analyzed=100,
        max_cols_analyzed=30,
        save_plot_dir=save_plot_dir
    )
    print("-"*100)

# 원천 데이터셋에 대한 데이터 품질 보고서 생성
def DQ_before_save(project,dataset):
    df = pd.read_csv(f'./ori_data/{dataset}.csv', encoding='utf-8')
    dq = data_cleaning_suggestions(df)

    dq = dq.to_csv(f'etc/{dataset}_DQ_before.csv')
    
    dq = pd.read_csv(f'etc/{dataset}_DQ_before.csv',encoding='utf-8')
    
    dq = dq.to_json(f'json/{dataset}_DQ_before.json', orient='records')
    
    # json 파일 불러오기
    with open(f'json/{dataset}_DQ_before.json', 'r') as file:
        json_data=file.read()
    json_api(project,'DQ_before',json_data)

# 데이터셋 전처리(기본)
def prepro_dataset(dataset, train_size,prediction):
    df = pd.read_csv(f'~/ori_data/{dataset}.csv', encoding='utf-8')
    ordinal_features = {col: df[col].unique() for col in df.columns if df[col].dtype == 'object'}
    su = setup(df, target = f'{prediction}', session_id = 123,train_size=train_size, max_encoding_ohe=2, ordinal_features=ordinal_features)
    #pycaret pull을 사용하여 이전 결과물 변수로 저장
    config = pull(su)
    # 아웃풋 데이터 프레임 형태로 변환
    config = pd.DataFrame(config)
    # json 형태로 변환
    config = config.to_json(orient='records')
    # json 파일 저장
    with open(f'json/{dataset}_config.json', 'w') as file:
        file.write(config)
    data = get_config('dataset_transformed')
    # 전처리 데이터 저장
    data.to_csv(f'prepro_data/{dataset}_transformed.csv',encoding='UTF-8',index=False)
    
# 성인 인구조사 데이터셋 전처리
def prepro_sample_adult(dataset,train_size,prediction):
    df = pd.read_csv(f'~/ori_data/{dataset}.csv', encoding='utf-8')
    su = setup(df, target = f'{prediction}', session_id = 123,train_size=train_size, max_encoding_ohe=2,
          ordinal_features={
              'workclass' : df['workclass'].unique(),
              'marital-status':df['marital-status'].unique(),
              'occupation' : df['occupation'].unique(),
              'relationship' : df['relationship'].unique(),
              'race' : df['race'].unique(),
              'native-country' : df['native-country'].unique()
          },
          numeric_imputation= 'knn',
          ignore_features='education',
          use_gpu=False,   
          feature_selection=False
    )
    #pycaret pull을 사용하여 이전 결과물 변수로 저장
    config = pull(su)
    # 아웃풋 데이터 프레임 형태로 변환
    config = pd.DataFrame(config)
    # json 형태로 변환
    config = config.to_json(orient='records')
    # json 파일 저장
    with open(f'json/{dataset}_config.json', 'w') as file:
        file.write(config)
    data = get_config('dataset_transformed')
    # 전처리 데이터 저장
    data.to_csv(f'prepro_data/{dataset}_transformed.csv',encoding='UTF-8',index=False)
# 대구 교통사고 예측 데이터셋 전처리
def prepro_sample_daegu(dataset,train_size,prediction):
    df = pd.read_csv(f'~/ori_data/{dataset}.csv', encoding='utf-8')
    df.columns = df.columns.str.replace(' ', '_')
    df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y-%m-%d %H')
    su = setup(df, target = f"{prediction}", session_id = 123,train_size=train_size,
               max_encoding_ohe=2,
               # date_features =['사고일시'],
               ordinal_features={'요일' : df['요일'].unique()},
                categorical_features=['기상상태', '시군구', '도로형태', '노면상태', '사고유형',
                                     '사고유형_-_세부분류', '법규위반', '가해운전자_차종',
                                     '가해운전자_성별', '가해운전자_연령', '가해운전자_상해정도',
                                     '피해운전자_차종', '피해운전자_성별', '피해운전자_연령',
                                     '피해운전자_상해정도'],
                numeric_imputation= 'knn',
                categorical_imputation = 'mode',
                remove_outliers=False,
             #   date_features = ['geton_date','geton_time','getoff_date', 'getoff_time'],
    
             
             ignore_features=['ID','사망자수','중상자수','경상자수','부상자수'],
              use_gpu=False,  
               normalize=False, 
              feature_selection=False
              # ,
                )
    #pycaret pull을 사용하여 이전 결과물 변수로 저장
    config = pull(su)
    # 아웃풋 데이터 프레임 형태로 변환
    config = pd.DataFrame(config)
    # json 형태로 변환
    config = config.to_json(orient='records')
    # json 파일 저장
    with open(f'json/{dataset}_config.json', 'w') as file:
        file.write(config)
    data = get_config('dataset_transformed')
    # 전처리 데이터 저장
    data.to_csv(f'prepro_data/{dataset}_transformed.csv',encoding='UTF-8',index=False)




# 전처리 후 데이터 시각화 코드
def lh_prepro_vis(dataset,prediction):
    try:
        df_prepro = pd.read_csv(f'prepro_data/{dataset}_transformed.csv', encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"'{dataset}_transformed.csv' 파일이 필요합니다. 파일명 끝에 '_transformed'를 붙여주세요.")
    print("\n전처리 데이터 시각화 1")
    walke2 = pyg.walk(df_prepro)
    with open(f"html/{dataset}_after.html", "w", encoding="utf-8") as f:
        f.write(pyg.to_html(df_prepro))
    print("-"*100)
    print("\n전처리 데이터 시각화 2")
    # bias_before
    AV = AutoViz_Class()
    # 시각화 결과 저장 경로
    save_plot_dir = f"image/after/"
    # 'html' 디렉토리가 없으면 생성
    os.makedirs('image/after', exist_ok=True)
    # 자동 시각화 실행
    dft = AV.AutoViz(
        filename="",
        sep=",",
        depVar=f"{prediction}",
        dfte=df_prepro,
        header=0,
        verbose=2,
        lowess=False,
        chart_format="png",
        max_rows_analyzed=100,
        max_cols_analyzed=30,
        save_plot_dir=save_plot_dir
    )
    print("-"*100)
# auto-viz 이미지 db 저장 코드
def lh_image_upload(pid, base_directory, prediction):
    if not os.path.exists(base_directory):
        raise FileNotFoundError(f" 주 파일경로 '{base_directory}'가 존재하지 않습니다 ")
    
    image_url = 'http://192.168.10.127:8000/postdb/light/insertFile/image/'
    
    # 테이블 이름과 해당하는 이미지 파일 이름을 매핑
    table_image_map = {
        "before": {
            "Scatter_Plots": "Scatter_Plots.png",
            "Pair_Scatter_Plots": "Pair_Scatter_Plots.png",
            "Heat_Maps": "Heat_Maps.png",
            "Dist_Plots_target": "Dist_Plots_target.png",
            "Dist_Plots_Numerics": "Dist_Plots_Numerics.png",
            "Box_Plots": "Box_Plots.png",
            "Bar_Plots": "Bar_Plots.png"
        },
        "after": {
            "Scatter_Plots_after": "Scatter_Plots.png",
            "Pair_Scatter_Plots_after": "Pair_Scatter_Plots.png",
            "Heat_Maps_after": "Heat_Maps.png",
            "Dist_Plots_target_after": "Dist_Plots_target.png",
            "Dist_Plots_Numerics_after": "Dist_Plots_Numerics.png",
            "Box_Plots_after": "Box_Plots.png"
        }
    }
    
    for directory in ["before", "after"]:
        dir_path = os.path.join(base_directory, directory, prediction)
        if not os.path.exists(dir_path):
            print(f"경고: '{dir_path}'라는 경로가 존재하지 않습니다")
            continue
        
        for table_name, image_file in table_image_map[directory].items():
            img_path = os.path.join(dir_path, image_file)
            
            if not os.path.exists(img_path):
                print(f"경고: {img_path} 파일이 없습니다. 건너뜁니다.")
                continue
            
            try:
                with open(img_path, 'rb') as img_file:
                    image_bytes = img_file.read()
                    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
                
                print(f"{image_file} 진행 중...")
                
                # API 요청 데이터 준비
                image_postVal = {
                    'val1': pid,  # 프로젝트 ID
                    'val2': table_name,  # 테이블명
                    'val3': encoded_string  # 텍스트로 변환된 이미지
                }
                
                 # API 요청 보내기
                response = requests.post(image_url, json=image_postVal)
                
                print(f"응답 코드: {response.status_code}")
                print(f"응답 내용: {response.text}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data is None:
                            print(f"경고: {table_name}로부터 응답이 없습니다")
                        else:
                            print(f"{table_name}으로 전송에 성공했습니다: {data}")
                    except json.JSONDecodeError:
                        print(f"경고: {table_name}에 대해 json 파싱에 실패했습니다")
                        print(f"응답 메시지: {response.text}")
                else:
                    print(f'{table_name}로부터 에러가 발생했습니다: {response.status_code}')
                    print(f'에러 메시지: {response.text}')
                    
            
            except FileNotFoundError:
                print(f"경고: {img_path} 파일이 없습니다.")
            except requests.exceptions.RequestException as e:
                print(f"{table_name}으로의 api 요청이 실패했습니다: {e}")
            except Exception as e:
                print(f"{image_file}을 저장하는데 예기치못한 오류가 발생했습니다.: {e}")
            print("\n","-"*100)



# 전처리 후 데이터 품질 보고서 생성
def DQ_after_save(project,dataset):
    df_prepro = pd.read_csv(f'prepro_data/{dataset}_transformed.csv',encoding='utf-8')
    dq = data_cleaning_suggestions(df_prepro)

    dq = dq.to_csv(f'etc/{dataset}_DQ_after.csv')
    
    dq = pd.read_csv(f'etc/{dataset}_DQ_after.csv',encoding='utf-8')
    
    dq = dq.to_json(f'json/{dataset}_DQ_after.json', orient='records')
    
    # json 파일 불러오기
    with open(f'json/{dataset}_DQ_after.json', 'r') as file:
        json_data=file.read()
    json_api(project,'DQ',json_data)
# 

      


# 모델 저장 코드
def model_saving(dataset,model,selected_model):
    print("\n모델 저장")
    with open(f'pkl/{dataset}_{model}.pkl', 'wb') as file:
        pickle.dump(selected_model, file)
    print(f"모델이 '{dataset}_{model}.pkl' 파일로 저장되었습니다.")

def performance_df_gen(project_type,model,y_test,y_pred):
    if project_type == '분류':
        metrics = {
            "Model": f"{model}",
            "Accuracy": [accuracy_score(y_test, y_pred)],
            "AUC": [roc_auc_score(y_test, y_pred)],
            "Recall": [recall_score(y_test, y_pred)],
            "Prec.": [precision_score(y_test, y_pred)],
            "F1": [f1_score(y_test, y_pred)],
            "Kappa": [cohen_kappa_score(y_test, y_pred)],
            "MCC": [matthews_corrcoef(y_test, y_pred)]
        }
    elif project_type == '회귀':
        metrics = {
            "Model": f"{model}",
            "MAE": [mean_absolute_error(y_test, y_pred)],
            "MSE": [mean_squared_error(y_test, y_pred)],
            "RMSE": [mean_squared_error(y_test, y_pred, squared=False)],
            "R2": [r2_score(y_test, y_pred)],
            "RMSLE": [np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))]
        }
    results_df = pd.DataFrame(metrics)
    print("Model Performance:")
    print(results_df)
    return results_df
        
    
# 모델 성능지표 시각화 코드
def model_vis(dataset, project_type, model):
    if project_type == '분류':
        with open(f'pkl/{dataset}_{model}.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        print("\n모델 성능 시각화")
        desired_directory = "image/performance/"
        # 디렉토리가 없으면 생성
        os.makedirs(desired_directory, exist_ok=True)
        
        # plot_model() 함수 사용 시 디렉토리 경로만 지정
        plot_model(loaded_model, plot='auc', save=desired_directory)
        plot_model(loaded_model, plot='confusion_matrix', save=desired_directory)
        plot_model(loaded_model, plot='threshold', save=desired_directory)
        plot_model(loaded_model, plot='pr', save=desired_directory)
        plot_model(loaded_model, plot = 'feature', save=desired_directory)
        # interpret_model(enhanced_model, X_new_sample=get_config('X_transformed')[:N], y_new_sample=get_config('y_transformed')[:N],save=desired_directory)
        # interpret_model(enhanced_model, save=desired_directory)
        # 파일 이름 변경 (필요한 경우)
        os.rename(os.path.join(desired_directory, "AUC.png"), os.path.join(desired_directory, f"{dataset}_AUC.png"))
        os.rename(os.path.join(desired_directory, "Confusion Matrix.png"), os.path.join(desired_directory, f"{dataset}_Confusion Matrix.png"))
        os.rename(os.path.join(desired_directory, "Threshold.png"), os.path.join(desired_directory, f"{dataset}_Threshold.png"))
        os.rename(os.path.join(desired_directory, "Precision Recall.png"), os.path.join(desired_directory, f"{dataset}_Precision Recall.png"))
        os.rename(os.path.join(desired_directory, "Feature Importance.png"), os.path.join(desired_directory, f"{dataset}_Feature Importance.png"))
        # os.rename(os.path.join(desired_directory, "SHAP summary.png"), os.path.join(desired_directory, f"{dataset}_SHAP summary.png"))
        print("\n","-"*100)
    elif project_type == '회귀':
        with open(f'pkl/{dataset}_{model}.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        print("\n모델 성능 시각화")
        desired_directory = "image/performance/"
        # 디렉토리가 없으면 생성
        os.makedirs(desired_directory, exist_ok=True)
        
        # plot_model() 함수 사용 시 디렉토리 경로만 지정
        plot_model(loaded_model, plot='residuals', save=desired_directory)
        plot_model(loaded_model, plot='error', save=desired_directory)
        plot_model(loaded_model, plot='cooks', save=desired_directory)
        plot_model(loaded_model, plot='learning', save=desired_directory)
        plot_model(loaded_model, plot = 'feature', save=desired_directory)
        # interpret_model(enhanced_model, X_new_sample=get_config('X_transformed')[:N], y_new_sample=get_config('y_transformed')[:N],save=desired_directory)
        # interpret_model(enhanced_model, save=desired_directory)
        # 파일 이름 변경 (필요한 경우)
        os.rename(os.path.join(desired_directory, "Residuals.png"), os.path.join(desired_directory, f"{dataset}_Residuals.png"))
        os.rename(os.path.join(desired_directory, "Prediction Error.png"), os.path.join(desired_directory, f"{dataset}_Prediction Error.png"))
        os.rename(os.path.join(desired_directory, "Cooks Distance.png"), os.path.join(desired_directory, f"{dataset}_Cooks Distance.png"))
        os.rename(os.path.join(desired_directory, "Learning Curve.png"), os.path.join(desired_directory, f"{dataset}_Learning Curve.png"))
        os.rename(os.path.join(desired_directory, "Feature Importance.png"), os.path.join(desired_directory, f"{dataset}_Feature Importance.png"))
        # os.rename(os.path.join(desired_directory, "SHAP summary.png"), os.path.join(desired_directory, f"{dataset}_SHAP summary.png"))
        print("\n","-"*100)
        

# 추천모델의 편향성 포함 성능지표(pycaret)
def lh_fairness(project,project_type,dataset,model,sensitive_features):
    #모델 편향성 체크
    print("\n모델 편향성 체크")
    # 저장한 모델 불러오기
    with open(f'pkl/{dataset}_{model}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    fairness = check_fairness(loaded_model, sensitive_features = sensitive_features)
    fair=pull()
    #json변환
    convert_pycaretfair_to_json(project_type, fair, dataset)
    with open(f'json/{dataset}_fairness.json', 'r') as file:
        json_data = file.read()
    # DB 저장 api 호출
    json_api(project,'et_fairness',json_data)

# 추천모델의 편향성 포함 성능지표2(pycaret)
def lh_custom_fairness(project_type,project,dataset, model, X, y, sensitive_features):
    # 모델 로드
    with open(f'pkl/{dataset}_{model}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        
    print(f"로드된 모델 타입: {type(loaded_model)}")  # 디버깅을 위한 출력
    
    if isinstance(loaded_model, str):
            # 만약 모델이 문자열이라면 적절한 모델 객체로 변환
            model_mapping = {
                'lr': LogisticRegression(random_state=42),
                'rf': RandomForestClassifier(random_state=42),
                'gbc': GradientBoostingClassifier(random_state=42),
                'ac': AdaBoostClassifier(random_state=42),
                'xgbc': XGBClassifier(random_state=42),
                'knc': KNeighborsClassifier()
                #필요한 모든 분류모델을 여기에 추가
            }
            loaded_model = model_mapping[model]
            print("모델을 문자열에서 객체로 변환했습니다.")
            
    # 데이터 준비
    data = pd.concat([X, y], axis=1)
    
    # PyCaret 실험 객체 생성 및 설정
    exp = ClassificationExperiment()
    exp = setup(data=data, target=y.name, session_id=42, verbose=False, 
                preprocess=False, 
                data_split_shuffle=False,
                data_split_stratify=False)
    
    # Fairness 검사 수행
    try:
        fairness_results = exp.check_fairness(
            estimator=loaded_model, 
            sensitive_features=sensitive_features
        )
    except AttributeError as e:
        print(f"오류 발생: {e}")
        print("PyCaret의 최신 버전에서 변경된 API를 사용해 보겠습니다.")
        fairness_results = check_fairness(
            estimator=loaded_model,
            sensitive_features=sensitive_features
        )
    fair=pull()
    convert_pycaretfair_to_json(project_type,fair, dataset)
    with open(f'json/{dataset}_fairness.json', 'r') as file:
        json_data = file.read()
    # DB 저장 api 호출
    json_api(project,'et_fairness',json_data)

# 모델 편향성 데이터프레임 생성(fairlearn)
def model_bias_check(project,project_type,dataset, sensitive_features,train_size,prediction,model):
    
        # 전처리된 데이터 불러오기
        df_prepro = pd.read_csv(f'prepro_data/{dataset}_transformed.csv', encoding='utf-8')
        # 예측값 제거
        X = df_prepro.drop([f'{prediction}'], axis=1)
        y = df_prepro[f'{prediction}']
        sensitive_group_all = df_prepro[sensitive_features]
        
        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
            X, y, sensitive_group_all, train_size=train_size, random_state=999)
        if project_type == '분류':
            # 저장한 모델 불러오기
            with open(f'pkl/{dataset}_{model}.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            # 임시 클로드 파이썬 파일
            constraint = DemographicParity()
            mitigator = ExponentiatedGradient(loaded_model, constraint)
            mitigator.fit(X_train, y_train, sensitive_features=X_train[sensitive_features])
            # 예측
            y_pred_original = loaded_model.predict(X_test)
            y_pred_mitigated = mitigator.predict(X_test)
            def calculate_metrics(y_true, y_pred, sensitive_features, col_nm='all'):
                return pd.DataFrame({
                    'date': date.today().strftime('%Y-%m-%d'),
                    'col_nm': col_nm,
                    'demographic_parity_difference': [demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)],
                    'demographic_parity_ratio': [demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)],
                    'equalized_odds_difference': [equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)],
                    'equalized_odds_ratio': [equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_features)]
                })
            
            # 원본 모델과 완화된 모델의 결과 비교
            fairlearn_data_original = []
            fairlearn_data_mitigated = []
            
            # 모든 특성을 고려한 공정성 지표 계산 ('all'로 설정)
            fairlearn_data_original.append(calculate_metrics(y_test, y_pred_original, A_test, col_nm='all'))
            fairlearn_data_mitigated.append(calculate_metrics(y_test, y_pred_mitigated, A_test, col_nm='all'))
            
            # 개별 민감한 특성에 대한 지표 계산
            for feature in sensitive_features:
                fairlearn_data_original.append(calculate_metrics(y_test, y_pred_original, A_test[[feature]], col_nm=feature))
                fairlearn_data_mitigated.append(calculate_metrics(y_test, y_pred_mitigated, A_test[[feature]], col_nm=feature))
            
            # 모든 결과를 각각 DataFrame으로 결합
            fairlearn_original = pd.concat(fairlearn_data_original, ignore_index=True)
            fairlearn_mitigated = pd.concat(fairlearn_data_mitigated, ignore_index=True)
        
            # JSON으로 변환 및 저장
            fairlearn_original_json = fairlearn_original.to_json(orient='records')
            fairlearn_mitigated_json = fairlearn_mitigated.to_json(orient='records')
        
            # API로 결과 전송
            json_api(project, 'fairlearn_before', fairlearn_original_json)
            json_api(project, 'fairlearn_after', fairlearn_mitigated_json)
        
            print("\n편향 전후 비교")
            print("\n편향 제거 이전:")
            print(fairlearn_original)
            print("\n", "-"*100)
            print("\n편향 제거 이후:")
            print(fairlearn_mitigated)
            
            # 설명 문자열 생성
            explanation = ("\ndemographic_parity_difference(인구 동등성 차이): 머신러닝 예측 확률이 민감 집단으로 인해 영향받지 않는 정도의 차이. 0에 가까울수록 적당\n" 
                           "demographic_parity_ratio(인구 동등성 비율): 머신러닝 예측 확률이 민감 집단으로 인해 영향받지 않는 정도의 차이. 1에 가까울수록 적당\n"
                           "equalized_odds_difference(균등화된 확률 차이): 예측 정도가 민감 집단에 영향받지 않는 정도는 물론, FP와 TP가 같은 수치를 보이고 있음을 나타내는 정도이다. 추가로 FP란 실제값이 거짓인데 참이라고 예측한 확률이며, TP는 실제값이 참이고, 참이라고 예측한 확률을 의미한다. 0에 가까울수록 모델이 더 공정. 0에서 멀어질수록 편향적.\n"
                           "equalized_odds_ratio(균등 기회 확률): 머신러닝 모델의 예측 정확도가 서로 다른 민감 집단 간에 얼마나 일관되는지를 나타내는 비율. 1에 가까울수록 모델이 더 공정하다고 판단. 1에서 크게 벗어난 값은 특정 집단에 대한 편향이 존재한다고 추정 가능")
            print(explanation)
            return fairlearn_mitigated




        elif project_type == '회귀':
            
                   # 저장한 모델 불러오기
            with open(f'pkl/{dataset}_{model}.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            
            # 손실 함수 클래스 정의
            class SquaredLoss:
                def __init__(self):
                    pass
        
                def eval(self, y_true, y_pred):
                    return (y_true - y_pred) ** 2
        
            # 상한값 계산
            y_pred = loaded_model.predict(X_train)
            squared_errors = (y_train - y_pred) ** 2
            upper_bound = np.mean(squared_errors) + 2 * np.std(squared_errors)
        
            # BoundedGroupLoss와 ExponentiatedGradient를 사용한 편향 완화
            constraint = BoundedGroupLoss(SquaredLoss(), upper_bound=upper_bound)
            mitigator = ExponentiatedGradient(estimator=loaded_model, constraints=constraint)
            mitigator.fit(X_train, y_train, sensitive_features=A_train)
            
            # 예측
            y_pred_original = loaded_model.predict(X_test)
            y_pred_mitigated = mitigator.predict(X_test)
            
            def calculate_metrics(y_true, y_pred, sensitive_features, col_nm='all'):
                group_min_metric = make_derived_metric(metric=mean_prediction, transform='group_min')
                group_max_metric = make_derived_metric(metric=mean_prediction, transform='group_max')
                
                group_mse = {}
                for feature in sensitive_features.columns:
                    group_mse[feature] = {}
                    for group in sensitive_features[feature].unique():
                        mask = sensitive_features[feature] == group
                        group_mse[feature][group] = mean_squared_error(y_true[mask], y_pred[mask])
                
                return pd.DataFrame({
                    'date': [date.today().strftime('%Y-%m-%d')],
                    'col_nm': [col_nm],
                    'mean_prediction': [mean_prediction(y_true, y_pred)],
                    'group_min_mean_prediction': [group_min_metric(y_true, y_pred, sensitive_features=sensitive_features)],
                    'group_max_mean_prediction': [group_max_metric(y_true, y_pred, sensitive_features=sensitive_features)],
                    'mean_prediction_gap': [group_max_metric(y_true, y_pred, sensitive_features=sensitive_features) - 
                                            group_min_metric(y_true, y_pred, sensitive_features=sensitive_features)],
                    'mse': [mean_squared_error(y_true, y_pred)],
                    'max_group_mse': [max(max(mse.values()) for mse in group_mse.values())]
                })
            
            # 원본 모델과 완화된 모델의 결과 비교
            fairlearn_data_original = []
            fairlearn_data_mitigated = []
            
            # 모든 특성을 고려한 공정성 지표 계산 ('all'로 설정)
            fairlearn_data_original.append(calculate_metrics(y_test, y_pred_original, A_test, col_nm='all'))
            fairlearn_data_mitigated.append(calculate_metrics(y_test, y_pred_mitigated, A_test, col_nm='all'))
            
            # 개별 민감한 특성에 대한 지표 계산
            for feature in sensitive_features:
                fairlearn_data_original.append(calculate_metrics(y_test, y_pred_original, A_test[[feature]], col_nm=feature))
                fairlearn_data_mitigated.append(calculate_metrics(y_test, y_pred_mitigated, A_test[[feature]], col_nm=feature))
            
            # 모든 결과를 각각 DataFrame으로 결합
            fairlearn_original = pd.concat(fairlearn_data_original, ignore_index=True)
            fairlearn_mitigated = pd.concat(fairlearn_data_mitigated, ignore_index=True)
        
            # JSON으로 변환 및 저장
            fairlearn_original_json = fairlearn_original.to_json(orient='records')
            fairlearn_mitigated_json = fairlearn_mitigated.to_json(orient='records')
        
            # API로 결과 전송
            json_api(project, 'fairlearn_before', fairlearn_original_json)
            json_api(project, 'fairlearn_after', fairlearn_mitigated_json)
        
            print("\n편향 전후 비교")
            print("\n편향 제거 이전:")
            print(fairlearn_original)
            print("\n", "-"*100)
            print("\n편향 제거 이후:")
            print(fairlearn_mitigated)
            
            # 설명 문자열 생성
            explanation = ("\nmean_prediction: 전체 데이터셋에 대한 평균 예측값\n"
                           "group_min_mean_prediction: 그룹별 평균 예측값 중 최소값\n"
                           "group_max_mean_prediction: 그룹별 평균 예측값 중 최대값\n"
                           "mean_prediction_gap: 그룹별 평균 예측값의 최대-최소 차이. 작을수록 더 공정\n"
                           "mse: 전체 데이터셋에 대한 평균 제곱 오차\n"
                           "max_group_mse: 모든 그룹 중 가장 큰 평균 제곱 오차. 작을수록 더 공정")
            print(explanation)
            
            return fairlearn_mitigated







# 시스템 편향 코드
def system_bias(pid, dataset, prediction, sensitive_features, train_size, model):
    df_prepro = pd.read_csv(f'prepro_data/{dataset}_transformed.csv', encoding='utf-8')
    N = 50
    X = df_prepro.drop(f'{prediction}', axis=1)[:N]
    y = df_prepro[f'{prediction}'][:N]
    # sensitive_features가 문자열이라면 리스트로 변환
    if isinstance(sensitive_features, str):
        sensitive_features = [sensitive_features]

    sensitive_group_all = df_prepro[sensitive_features][:N]  # 수정된 부분
    
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, sensitive_group_all, train_size=train_size, random_state=999)
    
    # 저장한 모델 불러오기
    with open(f'pkl/{dataset}_{model}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    
    # SHAP 값 계산
    explainer = shap.Explainer(loaded_model, X_train)
    shap_values = explainer(X_train)

    # 인덱스 접근 조정
    if len(shap_values.shape) == 2:  # (샘플 수, 피쳐 수)
        class_index = 0  # 첫 번째 클래스를 선택
        shap_value_to_plot = shap_values[0, :]  # 첫 번째 샘플의 SHAP 값
    elif len(shap_values.shape) == 3:  # (샘플 수, 피쳐 수, 클래스 수)
        class_index = 1  # 예: 두 번째 클래스를 선택
        shap_value_to_plot = shap_values[0, :, class_index]  # 첫 번째 샘플의 특정 클래스 SHAP 값
    else:
        raise ValueError("SHAP 값의 차원이 예상과 다릅니다.")
        
    # Waterfall 차트 생성
    image_path = os.path.abspath('image/performance')
    os.makedirs(image_path, exist_ok=True)  # 디렉토리가 없으면 생성
    file_path = os.path.join(image_path, f'{dataset}_waterfall.png')
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_value_to_plot, show=False)  # show=False로 설정
    plt.tight_layout()
    plt.draw()  # 그래프 렌더링
    print(f"Saving figure to: {file_path}")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()  # 메모리 해제
    
    print(f"Figure saved. File exists: {os.path.exists(file_path)}")
    
    # 저장된 이미지를 base64로 인코딩
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # DB에 저장
    image_url = 'http://192.168.10.127:8000/postdb/light/insertFile/image/'
    
    image_postVal = {
        'val1': pid,
        'val2': 'waterfall',
        'val3': encoded_image
    }
    try:
        response = requests.post(image_url, json=image_postVal)
        if response.status_code == 200:
            data = response.json()
            print(f"Waterfall 차트 저장 완료했습니다: {data}")
        else:
            print(f'Waterfall 차트 저장 중 오류가 발생했습니다: {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f"Waterfall 차트 API 요청이 실패했습니다: {e}")
    except Exception as e:
        print(f"Waterfall 차트를 저장하는데 예기치못한 오류가 발생했습니다.: {e}")




def confirm_log(pid,date, model, count, status):
    confirm_url = 'http://192.168.10.127:8000/postdb/light/insertMap/confirm/'
    confirm_postVal = {
            'val1': pid, # 프로젝트 ID
            'val2': 'string' , # confirm code?
            'val3': date,
            'val4': model,
            'val5': count,
            'val6': status,
            'val7': 'string',
        # confirm 데이터 or 텍스트로 변환된 이미지
    }
    response = requests.post(confirm_url,confirm = confirm_postVal)
    if response.status_code == 200:
        data = response.json()  # JSON 형식의 응답 데이터를 파이썬 딕셔너리로 변환
        print(f"Confirm id {data} 로 저장 완료됨")
    else:
        print(f'오류: {response.status_code}')
    


def train_log(pid,date, data, count, status):
    train_url = 'http://192.168.10.127:8000/postdb/light/insertMap/train/'
    train_postVal = {
            'val1': pid, # 프로젝트 ID
            'val2': 'string' , # train code?
            'val3': date,
            'val4': data,
            'val5': count,
            'val6': status,
            'val7': 'string',
        # train 데이터 or 텍스트로 변환된 이미지
    }
    response = requests.post(train_url,train = train_postVal)
    if response.status_code == 200:
        data = response.json()  # JSON 형식의 응답 데이터를 파이썬 딕셔너리로 변환
        print(f"Train id {data} 로 저장 완료됨")
    else:
        print(f'오류: {response.status_code}')
    


def lh_data_eda(project,dataset,prediction):
    # lh_ori_vis(dataset,prediction)
    # DQ_before_save(project,dataset)
    lh_prepro_vis(dataset,prediction)
    lh_image_upload(project,'image',prediction)
    DQ_after_save(project,dataset)



def lh_model_train_check_bias(project,project_type,dataset,train_size,prediction,model_pkl_path,model):
    # 훈련 평가 데이터셋 분할
    df = pd.read_csv(f'prepro_data/{dataset}_transformed.csv', encoding='utf-8')
    sensitive_features = [var for var in df.columns if df[var].nunique() <= 30 and var != f'{prediction}']
    X = df.drop(f'{prediction}',axis=1)
    y = df[f'{prediction}']
    X_train,X_test, y_train,y_test = train_test_split(X,y, train_size=train_size, random_state=999) 
    with open(model_pkl_path, 'rb') as file:
        loaded_model = pickle.load(file)
    y_pred = loaded_model.predict(X_test)
    if project_type == '분류':
        def calculate_metrics(y_true, y_pred, sensitive_features, col_nm='all'):
            return pd.DataFrame({
                'date': date.today().strftime('%Y-%m-%d'),
                'col_nm': col_nm,
                'demographic_parity_difference': [demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)],
                'demographic_parity_ratio': [demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)],
                'equalized_odds_difference': [equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)],
                'equalized_odds_ratio': [equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_features)]
            })
        metrics = {
            "Model": f"{model}",
            "Accuracy": [accuracy_score(y_test, y_pred)],
            "AUC": [roc_auc_score(y_test, y_pred)],
            "Recall": [recall_score(y_test, y_pred)],
            "Prec.": [precision_score(y_test, y_pred)],
            "F1": [f1_score(y_test, y_pred)],
            "Kappa": [cohen_kappa_score(y_test, y_pred)],
            "MCC": [matthews_corrcoef(y_test, y_pred)]
        }

        results_df = pd.DataFrame(metrics)
        print("Classification Model Performance:")
        print(results_df)
        convert_pycaretfair_to_json(project_type,results_df, dataset)

        # 분류 모델의 편향성 검사
        sensitive_group_all = df[sensitive_features]
        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
            X, y, sensitive_group_all, train_size=train_size, random_state=999)
        
        # 전체 sensitive features에 대한 단일 mitigator 생성
        constraint = DemographicParity()
        mitigator = ExponentiatedGradient(loaded_model, constraint)
        
        # 모든 sensitive features를 고려하여 mitigator 학습
        mitigator.fit(X_train, y_train, sensitive_features=A_train)
        
        # 편향이 제거된 모델 저장
        mitigated_model_path = f"{model_pkl_path.rsplit('.', 1)[0]}_mitigated.pkl"
        with open(mitigated_model_path, 'wb') as file:
            pickle.dump(mitigator, file)
        print(f"Saved mitigated model to {mitigated_model_path}")

        
        fairlearn_data_original = []
        fairlearn_data_mitigated = []
        
        # 각 컬럼별 편향성 검사
        for feature in sensitive_features:
            
            y_pred_original = loaded_model.predict(X_test)
            y_pred_mitigated = mitigator.predict(X_test)
            
            fairlearn_data_original.append(calculate_metrics(y_test, y_pred_original, A_test[feature], col_nm=feature))
            fairlearn_data_mitigated.append(calculate_metrics(y_test, y_pred_mitigated, A_test[feature], col_nm=feature))
        
        fairlearn_original = pd.concat(fairlearn_data_original, ignore_index=True)
        fairlearn_mitigated = pd.concat(fairlearn_data_mitigated, ignore_index=True)
        
        # JSON 변환 및 API 전송
        fairlearn_original_json = fairlearn_original.to_json(orient='records')
        fairlearn_mitigated_json = fairlearn_mitigated.to_json(orient='records')
        
        json_api(project, 'fairlearn_before', fairlearn_original_json)
        json_api(project, 'fairlearn_after', fairlearn_mitigated_json)
        
        print("\n분류 모델 편향성 분석 결과")
        print("\n편향 제거 이전:")
        print(fairlearn_original)
        print("\n", "-"*100)
        print("\n편향 제거 이후:")
        print(fairlearn_mitigated)
        
        print("\n분류 모델 편향성 지표 설명:")
        explanation = ("\ndemographic_parity_difference(인구 동등성 차이): 머신러닝 예측 확률이 민감 집단으로 인해 영향받지 않는 정도의 차이. 0에 가까울수록 적당\n" 
                      "demographic_parity_ratio(인구 동등성 비율): 머신러닝 예측 확률이 민감 집단으로 인해 영향받지 않는 정도의 차이. 1에 가까울수록 적당\n"
                      "equalized_odds_difference(균등화된 확률 차이): 예측 정도가 민감 집단에 영향받지 않는 정도는 물론, FP와 TP가 같은 수치를 보이고 있음을 나타내는 정도이다. 0에 가까울수록 모델이 더 공정\n"
                      "equalized_odds_ratio(균등 기회 확률): 머신러닝 모델의 예측 정확도가 서로 다른 민감 집단 간에 얼마나 일관되는지를 나타내는 비율. 1에 가까울수록 모델이 더 공정")
        print(explanation)
        
        return fairlearn_mitigated
    
    elif project_type == '회귀':
        def calculate_metrics(y_true, y_pred, sensitive_features, col_nm='all'):
            group_min_metric = make_derived_metric(metric=mean_prediction, transform='group_min')
            group_max_metric = make_derived_metric(metric=mean_prediction, transform='group_max')
            
            group_mse = {}
            for feature in sensitive_features.columns:
                group_mse[feature] = {}
                for group in sensitive_features[feature].unique():
                    mask = sensitive_features[feature] == group
                    group_mse[feature][group] = mean_squared_error(y_true[mask], y_pred[mask])
            
            return pd.DataFrame({
                'date': [date.today().strftime('%Y-%m-%d')],
                'col_nm': [col_nm],
                'mean_prediction': [mean_prediction(y_true, y_pred)],
                'group_min_mean_prediction': [group_min_metric(y_true, y_pred, sensitive_features=sensitive_features)],
                'group_max_mean_prediction': [group_max_metric(y_true, y_pred, sensitive_features=sensitive_features)],
                'mean_prediction_gap': [group_max_metric(y_true, y_pred, sensitive_features=sensitive_features) - 
                                      group_min_metric(y_true, y_pred, sensitive_features=sensitive_features)],
                'mse': [mean_squared_error(y_true, y_pred)],
                'max_group_mse': [max(max(mse.values()) for mse in group_mse.values())]
            })
        metrics = {
            "Model": f"{model}",
            "MAE": [mean_absolute_error(y_test, y_pred)],
            "MSE": [mean_squared_error(y_test, y_pred)],
            "RMSE": [mean_squared_error(y_test, y_pred, squared=False)],
            "R2": [r2_score(y_test, y_pred)],
            "RMSLE": [np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))]
        }
        results_df = pd.DataFrame(metrics)
        print("Model Performance:")
        print(results_df)
        convert_pycaretfair_to_json(project_type,results_df, dataset)
        
        # 손실 함수 클래스 정의
        class SquaredLoss:
            def __init__(self):
                pass

            def eval(self, y_true, y_pred):
                return (y_true - y_pred) ** 2

        # 상한값 계산
        y_pred = loaded_model.predict(X_train)
        squared_errors = (y_train - y_pred) ** 2
        upper_bound = np.mean(squared_errors) + 2 * np.std(squared_errors)

        # 데이터 준비
        sensitive_group_all = df[sensitive_features]
        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
            X, y, sensitive_group_all, train_size=train_size, random_state=999)

        # BoundedGroupLoss와 ExponentiatedGradient를 사용한 편향 완화
        constraint = BoundedGroupLoss(SquaredLoss(), upper_bound=upper_bound)
        mitigator = ExponentiatedGradient(estimator=loaded_model, constraints=constraint)
        
        # 모든 sensitive features를 고려하여 mitigator 학습
        mitigator.fit(X_train, y_train, sensitive_features=A_train)
        
        # 편향이 제거된 모델 저장
        mitigated_model_path = f"{model_pkl_path.rsplit('.', 1)[0]}_mitigated.pkl"
        with open(mitigated_model_path, 'wb') as file:
            pickle.dump(mitigator, file)
        print(f"Saved mitigated model to {mitigated_model_path}")
        
        fairlearn_data_original = []
        fairlearn_data_mitigated = []

        # 각 특성별로 개별적으로 편향성 검사
        for feature in sensitive_features:
            y_pred_original = loaded_model.predict(X_test)
            y_pred_mitigated = mitigator.predict(X_test)
            
            fairlearn_data_original.append(calculate_metrics(y_test, y_pred_original, A_test[[feature]], col_nm=feature))
            fairlearn_data_mitigated.append(calculate_metrics(y_test, y_pred_mitigated, A_test[[feature]], col_nm=feature))

        # 결과 DataFrame 생성
        fairlearn_original = pd.concat(fairlearn_data_original, ignore_index=True)
        fairlearn_mitigated = pd.concat(fairlearn_data_mitigated, ignore_index=True)

        # JSON으로 변환 및 API 전송
        fairlearn_original_json = fairlearn_original.to_json(orient='records')
        fairlearn_mitigated_json = fairlearn_mitigated.to_json(orient='records')

        json_api(project, 'fairlearn_before', fairlearn_original_json)
        json_api(project, 'fairlearn_after', fairlearn_mitigated_json)

        print("\n편향 전후 비교")
        print("\n편향 제거 이전:")
        print(fairlearn_original)
        print("\n", "-"*100)
        print("\n편향 제거 이후:")
        print(fairlearn_mitigated)
        
        explanation = ("\nmean_prediction: 전체 데이터셋에 대한 평균 예측값\n"
                      "group_min_mean_prediction: 그룹별 평균 예측값 중 최소값\n"
                      "group_max_mean_prediction: 그룹별 평균 예측값 중 최대값\n"
                      "mean_prediction_gap: 그룹별 평균 예측값의 최대-최소 차이. 작을수록 더 공정\n"
                      "mse: 전체 데이터셋에 대한 평균 제곱 오차\n"
                      "max_group_mse: 모든 그룹 중 가장 큰 평균 제곱 오차. 작을수록 더 공정")
        print(explanation)
        
        return fairlearn_mitigated