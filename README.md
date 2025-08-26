# traffic-imputation

Clone the repository:
git clone https://github.com/tamiewong/traffic-imputation.git

This project uses a Spatio-Temporal Graph Transformer to do traffic data imputation.

To set up
1. Navigate to desired location
2. Clone repo:
>> git clone https://github.com/<tamiewong>/<traffic-imputation>.git
3. Create virtual environment and install dependencies
>> cd <desired_location>
>> python -m venv myenv
>> myenv\Scripts\activate
>> python -m pip install --upgrade pip
>> pip install -r requirements.txt
>> pip install -e


To run the ST-GT
1. Open Windows PowerShell or Terminal
2. Activate environment
3. Navigate to the location of the repository
4. Run this line to train model:
python -m stgt.train --config configs/stgt.yaml
5. Run this line to evaluate model:
python -m stgt.run_eval --ckpt "outputs\stgt_YYYYMMDD-HHMMSS.pt"

Data size
Number of timestamps (temporal) = 2083
Number of roads/edges (spatial) = 63204
