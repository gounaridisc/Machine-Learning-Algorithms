# Credit Risk ML Algorithms

This folder contains a machine learning project built around **LendingClub credit risk prediction**. The work is organized as a set of Jupyter notebooks, each exploring a different algorithm on the same core dataset of borrower/application features available at or before loan issuance.

The main target is the multiclass label **`grade_idx` in {0..6}**, corresponding to LendingClub grades **A-G**. Most notebooks solve the full **7-class classification** problem; the MLP notebook additionally experiments with a simpler **3-bucket** version of the target.

## What We Built

We implemented and analyzed the following algorithms:

| Notebook | Type | What was built | Notes / reported result |
| --- | --- | --- | --- |
| `PCA.ipynb` | Dimensionality reduction | PCA from scratch with NumPy/SVD, train-only fitting, 2D/3D projections, explained-variance analysis, feature loadings | PC1 = 11.8%, PC2 = 10.0%; about 25 components are needed to retain ~95% variance |
| `KNN.ipynb` | Supervised classification | K-Nearest Neighbors from scratch using batched squared Euclidean distance and majority vote | Best model used **K = 10** with test accuracy **0.3094** |
| `Least Squares.ipynb` | Supervised classification | Multiclass least-squares classifier using one-hot targets and ridge-regularized normal equations | Test accuracy **0.3255** |
| `Logistic Regression.ipynb` | Supervised classification | Multiclass softmax regression from scratch with SGD, softmax, and cross-entropy | Test accuracy **0.3577** |
| `Naive Bayes.ipynb` | Supervised classification | Mixed Naive Bayes with Gaussian likelihoods for continuous features and categorical likelihoods with Laplace smoothing | Test accuracy **0.3037** |
| `SVM.ipynb` | Supervised classification | One-vs-Rest linear SVM from scratch with mini-batch SGD on hinge loss and a small dev-set hyperparameter search | Best tuned test accuracy **0.3017** |
| `K-Means.ipynb` | Unsupervised learning | K-Means clustering from scratch with multiple random initializations, inertia tracking, and post-hoc cluster-to-label mapping | Post-hoc test accuracy **0.2576** with **K = 7** |
| `MLP.ipynb` | Neural network classification | PyTorch MLP with multiple hidden layers, dropout, and training/evaluation curves | 3-bucket test accuracy **0.6823** |

## Dataset Summary

Across the notebooks, the dataset is treated consistently:

- **24 columns total**
- **2 label columns**: `grade`, `grade_idx`
- **22 input features**
- **4 categorical features**: `purpose`, `home_ownership`, `verification_status`, `addr_state`
- **18 numeric features** including `loan_amnt`, `annual_inc`, `dti`, `fico_avg`, `revol_util`, `open_acc`, and `total_acc`

Balanced splits are used throughout:

- **Train:** 59,619 rows
- **Dev:** 12,775 rows
- **Test:** 12,782 rows

Since the classes are balanced, a random 7-class classifier would be near **14.3% accuracy**, so all supervised models perform meaningfully above chance.

## Shared Pipeline Ideas

Even though each notebook is self-contained, the overall project follows the same modeling pattern:

1. Load train/dev/test CSV files.
2. Use only **pre-loan issuance** features.
3. Fit preprocessing on the **training split only** to avoid leakage.
4. One-hot encode categorical columns.
5. Standardize numeric features when appropriate.
6. Optionally apply `log1p` to highly skewed numeric variables.
7. Train the model.
8. Evaluate on train/dev/test with accuracy and, in several notebooks, confusion matrices and per-class analysis.

## What Each Notebook Contributes

### `PCA.ipynb`

This notebook is the project's exploratory dimensionality-reduction piece. PCA is implemented manually with SVD, then used to:

- visualize the training set in 2D and 3D,
- inspect explained variance,
- identify the strongest feature loadings behind principal components.

The results show that the dataset is not dominated by only a few directions: the first 10 components explain about **66.9%** of the variance, and around **25 components** are needed for ~95%.

### `KNN.ipynb`

This notebook implements KNN directly with NumPy. Distances are computed in batches for efficiency, and class prediction is done by majority vote among the nearest neighbors. Testing values from **K = 1 to 10** showed the expected pattern:

- `K = 1` overfits heavily,
- larger K values improve generalization,
- the best dev result appears at **K = 10**.

### `Least Squares.ipynb`

This notebook treats multiclass classification as a matrix regression problem by predicting one-hot targets and choosing the class with the maximum score. The model is trained with a ridge-regularized closed-form solution:

`W = (X^T X + lambda I)^(-1) X^T Y`

It gives stable but limited performance, with very similar train/dev/test accuracy, which suggests low variance and likely underfitting.

### `Logistic Regression.ipynb`

This notebook builds a full **multiclass softmax regression** model from scratch. It includes:

- manual softmax,
- manual cross-entropy loss,
- SGD training,
- train/dev learning curves,
- confusion matrix and per-class precision/recall/F1.

Among the 7-class classifiers in this folder, this is the strongest reported result. The model performs best on the extreme grades, especially **A** and **G**, and struggles most with middle grades where class overlap is stronger.

### `Naive Bayes.ipynb`

This notebook uses a **Mixed Naive Bayes** formulation because the dataset contains both numeric and categorical features:

- numeric columns are modeled with class-conditional Gaussians,
- categorical columns are modeled with discrete conditional probabilities,
- Laplace smoothing is used for robustness.

It is one of the simpler probabilistic baselines in the project.

### `SVM.ipynb`

This notebook implements a **linear One-vs-Rest SVM** from scratch using hinge loss and mini-batch SGD. It includes:

- a custom OvR training loop,
- objective tracking by epoch,
- a small hyperparameter search over `C` and learning rate.

The reported results suggest that the linear model is too simple for the full structure of this credit-grade task.

### `K-Means.ipynb`

This is the unsupervised part of the project. K-Means is implemented from scratch with:

- random centroid initialization,
- repeated restarts (`n_init`),
- inertia tracking,
- centroid updates,
- post-hoc majority-vote mapping from clusters to true labels.

Because labels are not used during training, the resulting alignment with grades is understandably weaker than the supervised models.

### `MLP.ipynb`

This notebook moves to a neural-network approach using **PyTorch**. Unlike the other notebooks, it groups the original 7 grades into **3 risk buckets**:

- **Low risk:** A-B
- **Medium risk:** C-D
- **High risk:** E-F-G

The network uses fully connected layers with ReLU activations and dropout. This is the best raw accuracy in the folder, but it is solving an easier **3-class bucketed problem**, so it should not be compared directly to the 7-class notebook results.

## High-Level Takeaways

- The folder covers a broad range of ML paradigms: **dimensionality reduction, instance-based learning, probabilistic modeling, linear models, margin-based learning, clustering, and neural networks**.
- Most of the classical algorithms are implemented **from scratch**, which makes this project useful not only for prediction, but also for learning the mechanics of each method.
- For the full **7-class** task, **logistic regression** appears to be the strongest reported supervised model in this folder.
- **K-Means** is useful here mainly as an exploratory unsupervised baseline rather than a strong predictive method.
- The **MLP** gives much higher accuracy, but on a coarser **3-bucket** target.

## Files In This Folder

- `PCA.ipynb`
- `KNN.ipynb`
- `Least Squares.ipynb`
- `Logistic Regression.ipynb`
- `Naive Bayes.ipynb`
- `SVM.ipynb`
- `K-Means.ipynb`
- `MLP.ipynb`
- `Data.pdf`

`Data.pdf` appears to be supporting project material for the dataset/assignment, while the core implementation work lives inside the notebooks.

## Running The Notebooks

These notebooks were authored in a **Google Colab + Google Drive** workflow. Several of them expect data at paths such as:

- `/content/drive/MyDrive/ML Project/Data/Train/train.csv`
- `/content/drive/MyDrive/ML Project/Data/Train/dev.csv`
- `/content/drive/MyDrive/ML Project/Data/Test/test.csv`

So if you want to rerun everything locally or in Colab, you will need access to those CSV files and may need to update the paths.

## Summary

In short, this folder documents a complete credit-risk modeling project built around LendingClub loan-grade prediction. It includes exploratory analysis, supervised learning baselines, an unsupervised clustering experiment, and a neural-network model, all centered on the same structured tabular dataset and evaluated on balanced train/dev/test splits.
