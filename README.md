# Algorithmic Accountability in Small Data: Sample-Size-Induced Bias in Classification Metrics

### Jarren Briscoe, Garrett Kepler, Daryl Deford, and Assefaw Gebremedhin

In the 28th International Conference on Artificial Intelligence and Statistics (AISTATS)

## Abstract
Evaluating machine learning models is crucial not only for determining their technical accuracy but also for assessing their potential societal implications. While the potential for low-sample-size bias in algorithms is well known, we demonstrate the significance of sample-size bias induced by combinatorics in classification metrics. This revelation challenges the efficacy of these metrics in assessing bias with high resolution, especially when comparing groups of disparate sizes, which frequently arise in social applications. We provide analyses of the bias that appears in several commonly applied metrics and propose a model-agnostic assessment and correction technique. Additionally, we analyze counts of undefined cases in metric calculations, which can lead to misleading evaluations if improperly handled. This work illuminates the previously unrecognized challenge of combinatorics and probability in standard evaluation practices, hoping to advance the community's approach for performing equitable and trustworthy classification methods.

## Reproducing Results: Sample-Size Induced Bias in Confusion-Matrix Metrics

To replicate the results from this paper, please ensure you are using Python 3.11. Follow the steps below:
1. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Execute the main script:
    ```bash
    python main.py
    ```
- If you encounter any issues during installation, a `pip_freeze.txt` file is available, which lists exact package versions. To use these, set up a clean Python 3.11 virtual environment in Windows 11 and run:
    ```bash
    pip install -r pip_freeze.txt
    ```
