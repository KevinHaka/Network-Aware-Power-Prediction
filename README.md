# Network-Aware Power Prediction

This project investigates whether the local position of a substation in a physical power network helps predict whether it will experience an interruption later. It compares a baseline logistic regression model using only substation-level attributes with a network-aware model that also includes graph-based features.

## Problem Statement

The goal is to predict whether an initially functioning substation will become interrupted later. The central question is whether information about a substation's position and connectivity within the power network improves prediction beyond traditional operational attributes such as load ratio and equipment age.

This is a binary classification problem:

- **0:** The substation is not interrupted later.
- **1:** The substation is interrupted later.

The analysis uses a synthetic grid-like network of substations and physical links. Although the data are synthetic, the workflow demonstrates how network science and machine learning can be combined for infrastructure-risk prediction.

## Dataset

The project uses two prepared CSV files:

- `power_substations.csv`: substation-level attributes and interruption outcomes.
- `power_links.csv`: physical connections between pairs of substations.

The substation dataset contains the following variables:

- `substation_id`: Unique substation identifier.
- `load_ratio`: Operational load ratio.
- `equipment_age_years`: Age of the equipment in years.
- `initially_interrupted`: Whether the substation was already interrupted at the prediction point.
- `interrupted_later`: Target variable indicating a later interruption.

The network contains **144 substations** and **344 physical links**. Substations that were initially interrupted are excluded from the prediction analysis, leaving **143 eligible substations**. Among them, **39 experienced a later interruption**.

## Methodology & Workflow

1. **Data Loading and Validation:**
   - Load the substation and link data from the published course data location.
   - Fall back to local CSV files when the online source is unavailable.
   - Validate identifiers, required columns, and network endpoints.

2. **Exploratory Data Analysis:**
   - Exclude substations that were initially interrupted.
   - Compare later interruption outcomes using a bar chart.
   - Use boxplots to compare `load_ratio` and `equipment_age_years` between substations that were and were not interrupted later.

3. **Network Construction:**
   - Build an undirected NetworkX graph from the physical links.
   - Add every substation as a node.
   - Calculate two local network features:
     - **Degree:** The number of direct physical connections of a substation.
     - **Local clustering coefficient:** The proportion of possible connections that exist between the substation's neighbours.

4. **Model Training:**
   - Use the same eligible observations and the same stratified 70/30 train-test split for both models.
   - Standardise the input features with `StandardScaler`.
   - Train logistic regression models with `max_iter=1000`.
   - **Model A — Original features:** `load_ratio` and `equipment_age_years`.
   - **Model B — Network-aware features:** Original features plus `degree` and `local_clustering`.

5. **Evaluation Strategy:**
   - Evaluate both models on the same held-out test set.
   - Report accuracy, F1-score, and ROC AUC.
   - Use confusion matrices to examine false positives and false negatives.

## Key Results

**Test set metrics (30% stratified hold-out set):**

| Model | Accuracy | F1-Score | ROC AUC |
|---|---:|---:|---:|
| Original substation features | 0.767 | 0.444 | 0.828 |
| Substation + network features | 0.860 | 0.750 | 0.949 |

The network-aware model substantially outperformed the baseline on this test split:

- Accuracy increased from **76.7% to 86.0%**.
- F1-score increased from **0.444 to 0.750**.
- ROC AUC increased from **0.828 to 0.949**.
- The network-aware model identified **9 later interruptions**, compared with only **4** for the baseline model.
- False negatives decreased from **8 to 3**, while false positives increased only slightly from **2 to 3**.

![Confusion Matrices With and Without Network Features](/CM_with_and_without_network_features.png)

![Network and Local Clustering Features](/network_and_lcc_degree_diagram.png)

![Exploratory Data Analysis Plots](/bar_and_box_plot.png)

## Interpretation

The results suggest that local network position contains useful predictive information. Two substations with similar load ratios and equipment ages may face different risks because their physical connectivity and neighbourhood structure differ.

Degree measures how many direct links a substation has, while local clustering measures how strongly its neighbours are connected to one another. Therefore, these features capture different aspects of local network redundancy and structure.

Adding network features helped the model detect later interruptions more effectively on the held-out test set. However, the results should be interpreted cautiously because the dataset is synthetic, relatively small, and evaluated using a single train-test split.

## Ethical Considerations

Using predictive models to prioritise infrastructure maintenance can have important social consequences. A **false negative** may result in a missed maintenance intervention and expose communities to unexpected power loss. A **false positive** may divert limited utility resources away from other substations.

For responsible deployment, model predictions should support rather than replace engineering judgement. The model should be evaluated on real and geographically diverse data, monitored for changes in performance, and audited by independent technical and regulatory bodies. Communities that depend on critical power infrastructure should also be represented in decisions about maintenance priorities and prediction thresholds.

## Reflection & Future Work

Possible extensions include:

- Evaluate the workflow on real utility outage and topology data.
- Use repeated stratified cross-validation instead of one train-test split.
- Tune the classification threshold to reduce costly false negatives.
- Compare logistic regression with Random Forest, XGBoost, SVM, or graph-based models.
- Add additional network features such as betweenness centrality, closeness centrality, and community membership.
- Investigate temporal network features and changing load conditions.
- Quantify uncertainty and provide confidence intervals for model performance.
- Test whether the model generalises to new network topologies and geographic regions.

## Repository Contents

- `Network Aware Power Prediction.ipynb`: Main analysis notebook.
- `Network Aware Power Prediction.html`: HTML export of the notebook.
- `Network Aware Power Prediction.pdf`: PDF report/export.
- `Synthetic_Power_Data_Construction.pdf`: Documentation about the construction of the synthetic power data.
- `CM_with_and_without_network_features.png`: Confusion-matrix comparison.
- `network_and_lcc_degree_diagram.png`: Network and local-feature visualisation.
- `bar_and_box_plot.png`: Exploratory data analysis plots.

## How to Run

1. Open `Network Aware Power Prediction.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run the cells from top to bottom.
3. If the online data source is unavailable, place `power_substations.csv` and `power_links.csv` in the same directory as the notebook.

The notebook uses Python and common data-science libraries, including:

- `pandas`
- `numpy`
- `matplotlib`
- `networkx`
- `scikit-learn`
