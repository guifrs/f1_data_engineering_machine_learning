from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from feature_engine import imputation
from sklearn import metrics, pipeline
from sklearn.ensemble import RandomForestClassifier
import bar_chart_race as bcr


import spark_ops

sns.set_theme()


def load_abt_from_spark(abt_path: str = "data/silver/abt_champions") -> pd.DataFrame:
    """
    Load the ABT (Analytical Base Table) from a Delta table into a Pandas DataFrame.

    Parameters
    ----------
    abt_path : str, optional
        Filesystem path to the ABT Delta table, by default "data/silver/abt_champions".

    Returns
    -------
    pd.DataFrame
        ABT data loaded into a Pandas DataFrame.
    """
    spark = spark_ops.new_spark_session()
    spark_ops.create_view_from_path(abt_path, spark)

    df = spark.table("abt_champions").toPandas()
    df["tempRoundNumber"] = df["tempRoundNumber"].astype(int)
    return df


def split_train_test_oot(
    df: pd.DataFrame,
    oot_year: int = 2024,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the ABT into train, test (by driver-year), and OOT (out-of-time) sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full ABT dataset.
    oot_year : int, optional
        Year to reserve as out-of-time set, by default 2024.
    test_ratio : float, optional
        Share of driver-year combinations to allocate to test, by default 0.2.
    random_state : int, optional
        Random seed for reproducible splits, by default 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_test, df_oot)
    """
    df_oot = df[df["dtYear"] == oot_year].copy()
    df_train_all = df[df["dtYear"] < oot_year].copy()

    # Driver-year level sampling
    df_sample = df_train_all[["DriverId", "dtYear"]].drop_duplicates().copy()

    rng = np.random.RandomState(random_state)
    df_sample["random"] = rng.uniform(size=df_sample.shape[0])
    df_sample["train"] = df_sample["random"] >= test_ratio
    df_sample["test"] = ~df_sample["train"]

    df_sample_train = df_sample[df_sample["train"]][["DriverId", "dtYear"]]
    df_sample_test = df_sample[df_sample["test"]][["DriverId", "dtYear"]]

    df_train = df_sample_train.merge(df_train_all, on=["DriverId", "dtYear"])
    df_test = df_sample_test.merge(df_train_all, on=["DriverId", "dtYear"])

    print("Total rows (pre train/test split):", df_train_all.shape[0])
    print("Train rows:", df_train.shape[0])
    print("Test rows:", df_test.shape[0])
    print("Train + test rows:", df_train.shape[0] + df_test.shape[0])

    return df_train, df_test, df_oot


def build_feature_sets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_oot: pd.DataFrame,
    target: str = "flChamp",
) -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]
]:
    """
    Build X/y for train, test and OOT sets from the given DataFrames.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset.
    df_test : pd.DataFrame
        Test dataset.
    df_oot : pd.DataFrame
        Out-of-time dataset.
    target : str, optional
        Target column name, by default "flChamp".

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]
        X_train, y_train, X_test, y_test, X_oot, y_oot, features
    """
    y_train = df_train[target]
    y_test = df_test[target]
    y_oot = df_oot[target]

    cols_to_remove = ["DriverId", "dtRef", target]
    features = [c for c in df_train.columns if c not in cols_to_remove]

    X_train = df_train[features]
    X_test = df_test[features]
    X_oot = df_oot[features]

    return X_train, y_train, X_test, y_test, X_oot, y_oot, features


def build_model_pipeline() -> pipeline.Pipeline:
    """
    Build the preprocessing + model pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with imputers and a RandomForest classifier.
    """
    features_imput_99 = [
        "avgPositionSprint",
        "avgGridPositionSprint",
        "medianPositionSprint",
        "medianGridPositionSprint",
        "avgPositionSprint1Year",
        "avgGridPositionSprint1Year",
        "medianPositionSprint1Year",
        "medianGridPositionSprint1Year",
        "avgPositionSprintCurrentTemp",
        "avgGridPositionSprintCurrentTemp",
        "medianPositionSprintCurrentTemp",
        "medianGridPositionCurrentTemp",
    ]

    imput_99 = imputation.ArbitraryNumberImputer(
        arbitrary_number=99,
        variables=features_imput_99,
    )

    features_imput_0 = [
        "avgPositionSprintGain",
        "medianPositionSprintGain",
        "avgPositionSprintGain1Year",
        "medianPositionSprintGain1Year",
        "avgPositionSprintGainCurrentTemp",
        "medianPositionSprintGainCurrentTemp",
    ]

    imput_0 = imputation.ArbitraryNumberImputer(
        arbitrary_number=0,
        variables=features_imput_0,
    )

    clf = RandomForestClassifier(
        random_state=42,
        min_samples_leaf=20,
    )

    return pipeline.Pipeline(
        steps=[
            ("imput_99", imput_99),
            ("imput_0", imput_0),
            ("random_forest", clf),
        ]
    )


def train_and_evaluate(
    model: pipeline.Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_oot: pd.DataFrame,
    y_oot: pd.Series,
) -> tuple[pipeline.Pipeline, np.ndarray]:
    """
    Train the model and compute metrics on train, test and OOT datasets.

    Parameters
    ----------
    model : pipeline.Pipeline
        Pipeline with preprocessing and estimator.
    X_train, y_train, X_test, y_test, X_oot, y_oot :
        Train, test and OOT feature matrices and target vectors.

    Returns
    -------
    Tuple[pipeline.Pipeline, np.ndarray]
        The fitted model and the predicted probabilities for the OOT set.
    """
    model.fit(X_train, y_train)

    # Train metrics
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    auc_train = metrics.roc_auc_score(y_train, y_prob_train)

    # Test metrics
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    acc_test = metrics.accuracy_score(y_test, y_pred_test)
    auc_test = metrics.roc_auc_score(y_test, y_prob_test)

    # OOT metrics
    y_pred_oot = model.predict(X_oot)
    y_prob_oot = model.predict_proba(X_oot)[:, 1]
    acc_oot = metrics.accuracy_score(y_oot, y_pred_oot)
    auc_oot = metrics.roc_auc_score(y_oot, y_prob_oot)

    print("=== Metrics ===")
    print(f"Train  - ACC: {acc_train:.3f} | AUC: {auc_train:.3f}")
    print(f"Test   - ACC: {acc_test:.3f} | AUC: {auc_test:.3f}")
    print(f"OOT    - ACC: {acc_oot:.3f} | AUC: {auc_oot:.3f}")

    # Feature importance
    transformed_cols = model[:-1].transform(X_train.head(1)).columns.tolist()
    feature_importance = pd.Series(
        model[-1].feature_importances_, index=transformed_cols
    ).sort_values(ascending=False)
    feature_importance = feature_importance[feature_importance > 0]

    print("\nTop feature importances:")
    print(feature_importance.head(20))

    return model, y_prob_oot


def plot_oot_top_drivers(df_oot: pd.DataFrame) -> None:
    """
    Plot the prediction trajectory over rounds for the top drivers in OOT.

    Parameters
    ----------
    df_oot : pd.DataFrame
        Out-of-time dataset with `predict` column already attached.
    """
    last_round = df_oot["tempRoundNumber"].max()
    top_drivers = (
        df_oot[df_oot["tempRoundNumber"] == last_round]
        .sort_values(by="predict", ascending=False)
        .head(5)["DriverId"]
        .unique()
        .tolist()
    )

    df_top_drivers = df_oot[df_oot["DriverId"].isin(top_drivers)].sort_values("dtRef")

    plt.figure(figsize=(9, 8), dpi=400)
    sns.lineplot(df_top_drivers, x="tempRoundNumber", y="predict", hue="DriverId")
    plt.title("Prediction over rounds (OOT - top drivers)")
    plt.savefig("figures/oot_top_drivers.png")


def score_future_season(
    model: pipeline.Pipeline,
    features: list[str],
    fs_path: str = "data/silver/feature_store_drivers",
    from_date: str = "2025-01-01",
) -> pd.DataFrame:
    """
    Score a future season using the feature store feature_store_drivers.

    Parameters
    ----------
    model : pipeline.Pipeline
        Fitted model pipeline.
    features : list of str
        List of feature column names used by the model.
    fs_path : str, optional
        Path to the feature_store_drivers Delta table, by default "data/silver/feature_store_drivers".
    from_date : str, optional
        Lower bound date (inclusive) for scoring, by default "2025-01-01".

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions for the future season.
    """
    spark = spark_ops.new_spark_session()
    spark_ops.create_view_from_path(fs_path, spark)

    df_future = (
        spark.table("feature_store_drivers").filter(f"dtRef > '{from_date}'").toPandas()
    )
    df_future["tempRoundNumber"] = df_future["tempRoundNumber"].astype(int)

    X_future = df_future[features]
    df_future["predict"] = model.predict_proba(X_future)[:, 1]

    return df_future


def plot_future_top5(df_future: pd.DataFrame) -> None:
    """
    Plot prediction curves for the top 5 drivers in the last round of the future season.

    Parameters
    ----------
    df_future : pd.DataFrame
        Future season DataFrame with `predict` column.
    """
    last_round = df_future["tempRoundNumber"].max()
    top_5 = (
        df_future[df_future["tempRoundNumber"] == last_round]
        .sort_values("predict", ascending=False)
        .head(5)["DriverId"]
        .tolist()
    )

    df_top_5 = (
        df_future[df_future["DriverId"].isin(top_5)]
        .copy()
        .sort_values("tempRoundNumber")
    )
    df_top_5["probability_pct"] = df_top_5["predict"] * 100

    plt.figure(figsize=(10, 5), dpi=300)
    bg_color = "#F5F5F5"
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    sns.lineplot(
        data=df_top_5,
        x="tempRoundNumber",
        y="probability_pct",
        hue="DriverId",
        linewidth=2,
        marker="o",
        markersize=5,
        palette="Set2",
    )

    plt.title(
        "Championship prediction over rounds – 2025 (Top 5)", fontsize=14, weight="bold"
    )
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Championship probability (%)", fontsize=12)

    plt.ylim(0, max(df_top_5["probability_pct"].max() * 1.1, 50))
    plt.xticks(df_top_5["tempRoundNumber"].unique())

    plt.legend(
        title="Driver",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig("figures/future_top5.png", dpi=300)
    plt.close()


def plot_combined_history(df_oot: pd.DataFrame, df_future: pd.DataFrame) -> None:
    """
    Plot combined prediction history (OOT + future season) for top drivers.

    Parameters
    ----------
    df_oot : pd.DataFrame
        Out-of-time dataset with predictions.
    df_future : pd.DataFrame
        Future season dataset with predictions.
    """
    columns_resume = [
        "tempRoundNumber",
        "DriverId",
        "dtRef",
        "dtYear",
        "predict",
        "qtdeWins1Year",
        "qtdePoles1Year",
        "medianPositionRaceCurrentTemp",
        "medianPosition1Year",
        "medianPositionRace1Year",
    ]

    last_round_oot = df_oot["tempRoundNumber"].max()
    top_drivers_oot = (
        df_oot[df_oot["tempRoundNumber"] == last_round_oot]
        .sort_values("predict", ascending=False)
        .head(5)["DriverId"]
        .unique()
        .tolist()
    )

    df_top_oot = df_oot[df_oot["DriverId"].isin(top_drivers_oot)][columns_resume]

    df_future_hist = df_future[columns_resume]
    df_hist = pd.concat([df_top_oot, df_future_hist]).sort_values("dtRef")

    plt.figure(figsize=(9, 8), dpi=400)
    sns.lineplot(df_hist, x="dtRef", y="predict", hue="DriverId", sort=True)
    plt.title("Prediction history (OOT + future season)")
    plt.savefig("figures/combined_history.png")


def make_bar_chart_race(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    value_col: str,
    output_path: str = "bar_chart_race.gif",
    n_bars: int = 5,
    title: str | None = None,
    figsize: tuple[int, int] = (14, 6),
    dpi: int = 180,
) -> Path:
    """
    Create an animated bar chart race GIF with a Formula 1 style visual identity.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = df[[time_col, entity_col, value_col]].copy()
    data = data.sort_values(by=[time_col, entity_col])
    data = data.drop_duplicates(subset=[time_col, entity_col], keep="last")
    data[value_col] = data[value_col] * 100

    wide_df = (
        data.pivot(index=time_col, columns=entity_col, values=value_col)
        .fillna(0)
        .sort_index()
    )

    bg_color = "#F2F2F2"
    font_color = "#000000"

    plt.rcParams.update(
        {
            "figure.facecolor": bg_color,
            "axes.facecolor": bg_color,
            "text.color": font_color,
            "axes.labelcolor": font_color,
            "xtick.color": font_color,
            "ytick.color": font_color,
        }
    )

    bcr.bar_chart_race(
        df=wide_df,
        filename=str(out_path),
        n_bars=n_bars,
        title=title,
        steps_per_period=18,
        period_length=1400,
        fixed_max=True,
        cmap="Reds_r",
        bar_label_size=16,
        tick_label_size=12,
        title_size=20,
        shared_fontdict={
            "family": "DejaVu Sans",
            "color": font_color,
            "weight": "bold",
        },
        figsize=figsize,
        dpi=dpi,
    )

    return out_path


def main() -> None:
    """
    End-to-end training script.

    Steps:
      1. Load ABT from Spark.
      2. Split into train / test / OOT.
      3. Build feature matrices and target vectors.
      4. Train model and print metrics.
      5. Score OOT and future season.
      6. Generate exploratory plots.
    """
    df_abt = load_abt_from_spark()

    df_train, df_test, df_oot = split_train_test_oot(df_abt, oot_year=2024)

    X_train, y_train, X_test, y_test, X_oot, y_oot, features = build_feature_sets(
        df_train, df_test, df_oot
    )

    model = build_model_pipeline()
    model, y_prob_oot = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, X_oot, y_oot
    )

    df_oot = df_oot.copy()
    df_oot["predict"] = y_prob_oot

    df_future = score_future_season(
        model, features, fs_path="data/silver/feature_store_drivers"
    )

    make_bar_chart_race(
        df=df_future,
        time_col="tempRoundNumber",
        entity_col="DriverId",
        value_col="predict",
        output_path="figures/future_bar_race.gif",
        n_bars=5,
        title="2025 Championship prediction (probability %)",
    )

    plot_oot_top_drivers(df_oot)
    plot_future_top5(df_future)
    plot_combined_history(df_oot, df_future)


if __name__ == "__main__":
    main()
