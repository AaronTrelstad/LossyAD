# LossyAD

A framework that combines [**TSB_AD**](https://github.com/TheDatumOrg/TSB-AD) with [**TerseTS**](https://github.com/cmcuza/TerseTS) to evaluate anomaly detection on compressed time series data.

## 🔗 Datasets

You can download the univariate time series anomaly datasets from the [TSB-AD-U](https://www.thedatum.org/datasets/TSB-AD-U.zip) collection and the multivariate time series anomaly datasets from the [TSB-AD-M](https://www.thedatum.org/datasets/TSB-AD-M.zip) collection.

**Instructions:**
1. Download the ZIP file from the link above.
2. Extract the contents.
3. Place the `TSB-AD-U` directory inside the `Datasets/` folder. (Should look like `Datasets/TSB-AD-U/...`)

## 📦 Build the project

Clone the repository:

```bash
git clone https://github.com/AaronTrelstad/LossyAD.git
cd LossyAD
```

Set up Conda environment:

```bash
conda create --name LossyAD python=3.10
conda activate LossyAD
```

Install Python requirements:

```bash
pip install -r requirements.txt
```

To install TerseTS:

```bash
git clone https://github.com/cmcuza/TerseTS.git
cd TerseTS/bindings/python
pip install .
```

## Usage

To run use:

```bash
python main.py
```