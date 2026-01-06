from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def train_and_export_model(output_path: str = "model.onnx") -> None:
    # 1. Tạo dữ liệu giả lập (Credit Risk: 4 features)
    # Feature 0: Income (scaled)
    # Feature 1: Debt (scaled) 
    # Feature 2: Age (scaled)
    # Feature 3: Credit History (scaled)
    X, y = make_classification(
        n_samples=1000, 
        n_features=4, 
        n_informative=3, 
        n_redundant=0, 
        random_state=42
    )
    
    # 2. Train model Logistic Regression
    clf = LogisticRegression()
    clf.fit(X, y)
    
    print("Model Coeffs:", clf.coef_)
    print("Model Intercept:", clf.intercept_)

    # 3. Convert sang ONNX
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    
    # 4. Save file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_and_export_model("models/credit_risk/v1/model.onnx")
