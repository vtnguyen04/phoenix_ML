from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def train_challenger_model(
    output_path: str = "models/credit_risk/v2/model.onnx"
) -> None:
    # Challenger Model: Train trên dữ liệu hơi khác một chút (giả lập data mới)
    # Tăng độ nhiễu (n_redundant) để model hoạt động khác đi
    X, y = make_classification(
        n_samples=1000, 
        n_features=4, 
        n_informative=3, 
        n_redundant=1, # Khác với v1 (0)
        random_state=99 # Seed khác
    )
    
    clf = LogisticRegression(C=0.5) # Regularization khác v1
    clf.fit(X, y)
    
    print(f"Challenger Coeffs: {clf.coef_}")

    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    print(f"Challenger Model saved to {output_path}")

if __name__ == "__main__":
    train_challenger_model()