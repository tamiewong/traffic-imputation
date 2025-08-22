# traffic-imputation

Clone the repository:
git clone https://github.com/tamiewong/traffic-imputation.git

This project uses a Spatio-Temporal Graph Transformer to do traffic data imputation.

To run the ST-GT
1. Open Windows PowerShell or Terminal
2. Activate environment
3. Navigate to the location of the repository
4. Run this line to train model:
python -m stgt.train --config configs/stgt.yaml
5. Run this line to evaluate model:
python -m stgt.run_eval

Data size
Number of timestamps (temporal) = 2083
Number of roads/edges (spatial) = 63204
