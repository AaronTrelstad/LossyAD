# LossyAD

A framework that combines [**TSB_AD**](https://github.com/TheDatumOrg/TSB-AD) with [**TerseTS**](https://github.com/cmcuza/TerseTS) to evaluate anomaly detection on compressed time series data.

## 🔗 Datasets

You can download the univariate time series anomaly datasets from the [TSB-AD-U](https://www.thedatum.org/datasets/TSB-AD-U.zip) collection.

**Instructions:**
1. Download the ZIP file from the link above.
2. Extract the contents.
3. Place the `TSB-AD-U` directory inside the `Datasets/` folder.

## 📦 Installation

To install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run use:

```bash
python -m TSB_AD.main
```

Note: Before re-running experiments, delete the eval/ directory.


