import pandas as pd
import lighthouse_func as lf
import lh_variable as lv
import time

 
# 변수 불러오기
config = lv.get_project_config()




# %%time
# 문제 유형 받기
lf.lighthouse_setup(config.project_type)

# %%time
# %%capture
# 전처리 데이터 시각화, 시각화 DB 업로드, 데이터 품질 보고서 저장
lf.lh_data_eda(config.project,config.dataset,config.prediction)


# %%time
#모델 편향 체크
start = time.time()
lf.lh_model_train_check_bias(config.project,config.project_type,config.dataset,config.train_size,config.prediction,config.model_pkl_path,config.model_nm)
end = time.time()
print(f"{end - start:.5f} sec")

# 범주형 유니크값 제한하기(특정 개수 이하로.)