# traffic-imputation

This project uses a **Spatio-Temporal Graph Transformer (ST-GT)** for traffic data imputation.

## Clone the repository
```
git clone https://github.com/tamiewong/traffic-imputation.git
```

## Setup

1. **Navigate to the cloned repository**
   ```
   cd traffic-imputation
   ```

2. **Create a virtual environment and install dependencies**
   ```
   python -m venv myenv
   myenv\Scripts\activate      # On Windows
   # source myenv/bin/activate  # On macOS/Linux

   python -m pip install --upgrade pip
   ```

   - **If using CPU:**
     ```
     pip install -r requirements.txt
     pip install -e
     ```

   - **If using GPU (Nvidia CUDA):**
     ```
     pip install -r requirements_GPU.txt
     pip install -e
     ```

3. **Prepare the data**
   - Ensure the following file exists after extraction/placement:
     ```
     data/utd_agg.parquet
     ```

4. **Configuration**
   - If using CPU, set `use_cuda: False` in `configs/stgt.yaml`.

## Running the Model

1. Open PowerShell (Windows) or Terminal (macOS/Linux).
2. Activate your environment:
   ```
   myenv\Scripts\activate      # On Windows
   # source myenv/bin/activate  # On macOS/Linux
   ```
3. Navigate to the repository folder:
   ```
   cd traffic-imputation
   ```
4. Train the model:
   ```
   python -m stgt.train --config configs/stgt.yaml
   ```
5. Model performance will be printed in the terminal at the end of training.

## Data Information

- **Number of timestamps (temporal):** 2083
- **Number of roads/edges (spatial):** 63,204
