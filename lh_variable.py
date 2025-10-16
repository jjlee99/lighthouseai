from dataclasses import dataclass
from typing import Dict, Any
import json


#분류/회귀
# project = '29' # 프로젝트 번호를 입력해주세요.
# dataset = 'adult-sklearn' # 다운받으신 데이터셋명을 확장자 제외하고 입력해주세요.
# train_size = 0.7 # 훈련-평가 데이터셋 비율입력하십시오.
# prediction = 'income' # 예측하려는 변수를 입력하세요.
# project_type = '분류' # 머신러닝 방법(회귀, 분류)은 선택해서 입력해주십시오.
# model_nm = 'gbc' # 사용하고자하는 머신러닝 모델을 입력하세요. 
# model_pkl_path = 'pkl/adult-sklearn_gbc.pkl' # 현재 보유하고 있는 머신러닝 모델 파일 경로(.pkl)
# # (rf : 랜덤포레스트, lr : 선형 분류 모델, xgboost : 극한 변화도 부스팅 모델, gbc : 그래디언트 부스팅 모델, et: 엑스트라 트리 모델 등등)
# # sensitive_features = ['gender','race'] # 모든 컬럼으로 설정.
# verboose = False # 아웃풋을 생략할지 표기할지 설정합니다

@dataclass
class ProjectConfig:
    project: str
    dataset: str
    train_size: float
    model_nm: str
    project_type: str
    model_pkl_path: str
    verbose: bool
    prediction : str
    
    def to_dict(self) -> Dict[str, Any]:
        """설정값들을 dictionary 형태로 변환합니다."""
        return {
            "project": self.project,
            "dataset": self.dataset,
            "train_size": self.train_size,
            "model_nm": self.model_name,
            "project_type": self.project_type,
            "model_pkl_path": self.model_pkl_path,
            "verbose": self.verbose,
            "prediction" : self.prediction
        }
    
    def to_json(self) -> str:
        """설정값들을 JSON 문자열로 변환합니다."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProjectConfig':
        """dictionary로부터 ProjectConfig 객체를 생성합니다."""
        return cls(**config_dict)

def get_project_config() -> ProjectConfig:
    """프로젝트 기본 설정값을 반환합니다."""
    return ProjectConfig(
        project='29',  # 프로젝트 번호를 입력하세요.
        dataset='adult-sklearn',  # 다이렉션의 데이터셋명 확장자 제외하고 입력하주세요.
        train_size=0.7, # 훈련용 데이터셋 비율 입력하세요.
        prediction = 'income',
        model_nm='gbc',  # 사용하고자하는 머신러닝 모델을 입력하세요.
        project_type='분류',  # 머신러닝 방법(회귀, 분류)은 선택해서 입력하세시오.
        model_pkl_path='pkl/adult-sklearn_gbc.pkl',  # 현재 보유하고 있는 머신러닝 모델 파일 경로
        verbose=False  # 아웃풋을 상세하게 표기할지 설정합니다.
    )

# 사용 예시
if __name__ == "__main__":
    # 설정 가져오기
    config = get_project_config()
    
    # JSON 형태로 출력 (프론트엔드로 전달 가능)
    print(config.to_json())
    
    # Dictionary 형태로 출력
    print(config.to_dict())