import numpy as np
import pandas as pd
from scipy.stats import loguniform
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import  plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


# VISUALIZATIONS


def plot_countplot(df: pd.DataFrame, feature1: str, feature2: str, title: str) -> None:
    """Takes as an input name of the pd.DataFrame and names of needed columns and plots a count plot.

        :param: df: the name of the pd.DataFrame to use;
                feature1: str - name of the columns to plot on X axis;
                feature2: str - name of the columns to plot as a 'hue' parameter (to do some cross counting);
                title: str -  final title (name) of the whole plot.
        """

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(x=feature1, hue=feature2, data=df, palette="viridis")

    if feature2 != None:
        ax.legend(loc="upper right", title=feature2)
        ax.bar_label(ax.containers[1])

    ax.set_title(title)
    ax.set(ylabel="")
    ax.set(xlabel=feature1)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)
    plt.tight_layout()


def plot_line_plot_plotly(df: pd.DataFrame, x: str, y: str, z: str, title: str) -> None:
    """Takes as input name of the pd.DataFrame, names of columns ant plots a line plot.
        
        param: df:  the name of the pd.DataFrame to use;
               x: str - name of the column to plot on X axis;
               y: str - name of the column to plot on Y axis and to name the points on the line plot;
               z: str - name of the column to plot as hue (color) to the different lines;
               title: str - the title of the whole plot.
        """
    fig = px.line(df, x=x, y=y, color=z, text=y,)
    fig.update_traces(textposition="top left")
    fig.update_layout(legend_title="", title=title)
    fig.show()


def plot_stacked_barchart_plotly(
    df: pd.DataFrame, x: str, y: list, title: str, legend_title: str
) -> None:
    """Takes as an input name of the pd.Dataframe, needed columns, titles and plots a stacked bar chart of percentage
        
        param: df: the name of the pd.DataFrame to use;
               x: str - name of the column to plot on X axis;
               y: str - list of names (str) of the columns with percentage to plot on Y axis;
               title: str - title of the whole chart;
               legend_title: str - to rename the legend.
        """
    fig = px.bar(
        df,
        x=x,
        y=y,
        text_auto=True,
        color_discrete_map={y[0]: "#26828e", y[1]: "#35b779"},
        title=title,
    )
    fig.update_yaxes(title="")
    fig.update_xaxes(title="")
    fig.update_layout(legend_title=legend_title)
    fig.show()


def plot_box_stripplot(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    """Takes as an input name of the pd.DataFrame, certain columns to plot on axis and plots boxplot+stripplot together.

        :param: df: the name of the pd.DataFrame to use;
               x: str - name of the column to plot on X axis;
               y: str - name of the column to plot on Y axis;
               title: str - title of the whole chart;
        """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_yscale("linear")
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set(xlabel="")
    sns.boxplot(x=x, y=y, data=df, palette="viridis")

    sns.stripplot(x=x, y=y, data=df, palette="GnBu", size=5, edgecolor="gray")

    sns.despine(trim=True, left=True)
    ax.set_title(title)


def plot_heatmap(df: pd.DataFrame, title: str) -> None:
    """Takes as an input pd.DataFrame and plot the heatmap with correlation coefficients between all features.

        :param: df: the name of the pd.DataFrame to use;
                title: str - title of the whole heatmap.
        """
    sns.set_theme(style="white")
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(10, 8))

    cmap = sns.diverging_palette(150, 275, as_cmap=True)
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
    )

    heatmap.set_title(
        title, fontdict={"fontsize": 16}, pad=12,
    )
    plt.xlabel("")
    plt.ylabel("")


# COUNTING


def make_crosstab_number(
    df: pd.DataFrame, feature1: str, feature2: str
) -> pd.DataFrame:
    """Takes as an input name of the pd.DataFrame and certain features to use, 
    outputs Pd.DataFrame with cross count of the values in these features.

    :param: df: the name of the pd.DataFrame to use;
            feature1: str - name of the first column which values to cross count;
            feature2: str - name of the second column which values to cross count;
    :return: pd.DataFrame with statistics of cross counted values from both used columns.
    """
    return pd.crosstab(df[feature1], df[feature2])


def make_crosstab_percent(
    df: pd.DataFrame, feature1: str, feature2: str
) -> pd.DataFrame:
    """Takes as an input name of the pd.DataFrame and certain features to use, 
    outputs Pd.DataFrame with cross count and turn into percent of the values in these features.

    :param: df: the name of the pd.DataFrame to use;
            feature1: str - name of the first column which values to cross count;
            feature2: str - name of the second column which values to cross count;
    :return: pd.DataFrame with percent of cross counted values from both used columns.
    """
    return pd.crosstab(df[feature1], df[feature2], normalize="index") * 100


# MODELS


def calc_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame (without dependent variable) as input and returns coefficients of Variance Inflation factor
        in the form of od.DataFrame.

        :params: df - pd.Dataframe - as X_train with added constant form, without dependent variable.
        return: pd.DataFrame with all column names (as features) and VIF coefficients.
        """
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif


def base_line(X: pd.DataFrame, y: pd.DataFrame, preprocessor: np.array) -> pd.DataFrame:
    """
        Takes as an input X (all usable predictors) and y (outcome, dependent variable) pd.DataFrames.
        The the function performs cross validation with different already selected models.
        Returns metrics and results of the models in pd.DataFrame format.

        :param: X - pd.DataFrame of predictors(independent features);
                y - pd.DataFrame of the outcome;
                preprocessor: ColumnTransformer with all needed scalers, transformers.
        """
    balanced_accuracy = []
    roc_auc = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    fit_time = []
    kfold = StratifiedKFold(n_splits=5)
    classifiers = [
        "Logistic regression",
        "Decision Tree",
        "Random Forest",
        "Linear SVC",
        "SVC",
        "KNN",
    ]

    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        LinearSVC(),
        SVC(),
        KNeighborsClassifier(),
    ]

    for model in models:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model),]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=kfold,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
                "roc_auc",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        balanced_accuracy.append(result["test_balanced_accuracy"].mean())
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())
        roc_auc.append(result["test_roc_auc"].mean())
    base_models = pd.DataFrame(
        {
            "Balanced accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Roc Auc": roc_auc,
            "Fit_time": fit_time,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="YlGn")
    return base_models


def plot_classifier_scores(
    model, X: pd.DataFrame, y: pd.DataFrame, predictions: np.array
) -> None:
    """Plots the Confusion matrix and classification report from scikit-learn.
        
        :param: model - chosen model, modeled Pipeline from sklearn, on which data is trained.
                X - pd.DataFrame, X_train, X_validation, X_test data, which on to predict and plot the prediction 
                result.
                y - pd.DataFrame, the outcome, dependent variable: y_train. y_val, y_test, what to predict.
                predictions: y_hat, predictions from the model.
        """
    cmap = sns.diverging_palette(150, 275, as_cmap=True)
    plot_confusion_matrix(model, X, y, cmap=cmap)
    plt.title("Confusion Matrix: ")
    plt.show()
    print(classification_report(y, predictions))

    print()


def KNN_objective(
    trial, X: pd.DataFrame, y: pd.DataFrame, numeric_features: list
) -> float:
    """Takes as an input pd.DataFrames with features and outcome, gives the best score of f1 after training and
        cross validation.

        :params: trial : a process of evaluating an objective function;
                X: pd.DataFrame with independent features (predictors);
                y: pd.DataFrame with the outcome (what to predict);
                numeric_features: list - names of the features, which must be scaled with scaler (numerical columns);
        :returns: the score, this time - f1 - after fitting data with different hyper parameters to the model and
        cross validation.
        """
    # (a) List scalers to chose from
    scalers = trial.suggest_categorical("scalers", ["minmax", "standard", "robust"])

    # (b) Define your scalers
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    # (a) List all dimensionality reduction options
    dim_red = trial.suggest_categorical("dim_red", ["PCA", None])

    # (b) Define the PCA algorithm and its hyperparameters
    if dim_red == "PCA":
        # suggest an integer from 2 to 10 (as in total now I have 11 features)
        pca_n_components = trial.suggest_int("pca_n_components", 2, 10)
        dimen_red_algorithm = PCA(n_components=pca_n_components)
    # (c) No dimensionality reduction option
    else:
        dimen_red_algorithm = "passthrough"

    # -- Instantiate estimator model
    knn_n_neighbors = trial.suggest_int("knn_n_neighbors", 1, 10, 1)
    knn_metric = trial.suggest_categorical(
        "knn_metric", ["euclidean", "manhattan", "minkowski"]
    )
    knn_weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])

    estimator = KNeighborsClassifier(
        n_neighbors=knn_n_neighbors, metric=knn_metric, weights=knn_weights
    )

    # -- Make a pipeline
    preprocessor = ColumnTransformer(
        transformers=[("scaler", scaler, numeric_features)], remainder="passthrough"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("reduction", dimen_red_algorithm),
            ("estimator", estimator),
        ]
    )

    # -- Evaluate the score by cross-validation
    score = cross_val_score(pipeline, X, y, scoring="f1")
    f1 = score.mean()
    return f1


def random_forest_objective(trial, X: pd.DataFrame, y: pd.DataFrame) -> float:
    """ Takes as an input training and outcome pd.DataFrame's, gives the best score of f1 after training and
        cross validation.
        :param: trial a process of evaluating an objective function;
                X: pd.DataFrame with independent features (predictors);
                y: pd.DataFrame with the outcome (what to predict);
        :returns: the score, this time - f1 - after fitting data with different hyper parameters to the model and
        cross validation.
        """
    _n_estimators = trial.suggest_int("n_estimators", 20, 1000)
    _max_depth = trial.suggest_int("max_depth", 2, X.shape[1])
    _min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
    _max_features = trial.suggest_int("max_features", 2, 10)
    _class_weight = trial.suggest_categorical(
        "weight", ["balanced", "balanced_subsample", None]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    model = RandomForestClassifier(
        criterion="gini",
        n_estimators=_n_estimators,
        max_depth=_max_depth,
        min_samples_split=_min_samples_split,
        min_samples_leaf=_min_samples_leaf,
        max_features=_max_features,
        class_weight=_class_weight,
    )

    score = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
    return score


def SVC_randomized_search(X: pd.DataFrame, y: pd.DataFrame, model: Pipeline) -> None:
    """SVC hyper parameter searcher.
        
        Takes as an input X (independent variables(outcome) pd.DataFrame and a Pipeline with 
        preprocessors, transformers and certain model(SVC), fits the given data and searches for the best 
        hyper parameters.

        :param: X: pd.DataFrame with features;
                y: pd.DataFrame with outcome (dependent variable);
                model: sklearn.Pipeline with all needed transformers, preprocessors and chosen main model, in this
                example - SVC.
        """
    param_grid = [
        {
            "classifier__C": loguniform(1e-5, 100),
            "classifier__kernel": ["rbf"],
            "classifier__gamma": [10, 1, 0.1, 0.01],
            "classifier__class_weight": ["balanced", None],
            "classifier": [SVC()],
        },
    ]

    search = RandomizedSearchCV(
        model, param_grid, scoring="f1", n_iter=20, n_jobs=-1, cv=5, random_state=123
    )

    result = search.fit(X, y)

    print(f"Best params:")
    print(result.best_params_)
    print("Best f1 score in randomized search:")
    print(result.best_score_)