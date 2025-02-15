# Python 3 AI Trader Predictor
A trading AI assistant to help with your trades.

## Installation
1. Clone repo
```
git clone https://github.com/sudowoodo-keym/trading_ai.git
```
2. Create venv
```
python3 -m venv venv
```
3. Activate venv
```
source /venvbin/activate
```
4. Install dependencies
```
pip install -r dependencies.txt
```
5. Run program
```
python3 quant_trading.py
```
## Configuration
Adjust the hyperparameters to your specifications, since this uses scikit-learn it mainly uses the processor.
```
 param_grid = {
            'n_estimators': [150, 250, 500], # Reduce total trees for faster training
            'max_depth': [5, 10], # Limit tree depth to prevent excessive memory usage
            'min_samples_split': [2, 4], # Prevent very deep splits
            'min_samples_leaf': [1, 2, 4], # Ensures each leaf has enough data points
            'max_features': ['sqrt'] # Limit feature selection per split
        }
```
