import pyspark
from pyspark.sql import DataFrame, SparkSession
from delta import DeltaTable, configure_spark_with_delta_pip
from rich.progress import track
from pathlib import Path


def read_query(path: str) -> str:
    """
    Read a SQL query from a given file path.

    Parameters
    ----------
    path : str
        Path to the file containing the SQL query.

    Returns
    -------
    str
        The content of the file as a single SQL string.
    """
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def new_spark_session() -> SparkSession:
    """
    Create and configure a SparkSession with Delta Lake support.

    Returns
    -------
    SparkSession
        An active SparkSession instance configured for Delta Lake.
    """
    builder = (
        pyspark.sql.SparkSession.builder.appName("MyApp")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.debug.maxToStringFields", "10000")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark


def create_view_from_path(path: str, spark: SparkSession) -> None:
    """
    Create (or replace) a temporary view from a Delta table path.

    The view name is derived from the last segment of the given path.

    Parameters
    ----------
    path : str
        Filesystem path to the Delta table.
    spark : SparkSession
        Active SparkSession used to read and register the view.
    """
    df = spark.read.format("delta").load(path)
    table_name = Path(path).name
    df.createOrReplaceTempView(table_name)


def create_table(query_path: str, spark: SparkSession) -> None:
    """
    Create or overwrite a Delta table from a SQL file.

    The table name and output path are derived from the query file name.
    The result is written to the `data/silver/<table>` folder and then
    vacuumed to clean up old files.

    Parameters
    ----------
    query_path : str
        Path to the file containing the SQL query.
    spark : SparkSession
        Active SparkSession used to execute the query and write the table.
    """
    table_name = Path(query_path).stem
    query = read_query(query_path)

    df = spark.sql(query)

    (
        df.coalesce(1)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"data/silver/{table_name}")
    )

    delta_table = DeltaTable.forPath(spark, f"data/silver/{table_name}")
    delta_table.vacuum()


class IngestorFS:
    """
    File-based incremental ingestor for Delta Lake tables.

    This class reads a SQL template from disk, executes it for a series of
    reference dates, and writes/updates a partitioned Delta table on the
    filesystem.

    The SQL template is expected to contain a `{date}` placeholder that
    will be formatted with each iteration value.

    Attributes
    ----------
    table : str
        Logical table name derived from the query file name.
    spark : SparkSession
        Active SparkSession used to run queries and write data.
    query : str
        SQL template loaded from the query file path.
    """

    def __init__(self, query_path: str, spark: SparkSession) -> None:
        """
        Initialize the ingestor with a SQL file and a SparkSession.

        Parameters
        ----------
        query_path : str
            Path to the SQL template file. Must contain a `{date}` placeholder.
        spark : SparkSession
            Active SparkSession used to execute the queries.
        """
        self.table = Path(query_path).stem
        self.spark = spark
        self.query = read_query(query_path)

    def load(self, date: str) -> DataFrame:
        """
        Load data for a given reference date by executing the SQL template.

        Parameters
        ----------
        date : str
            Reference date to inject into the SQL template, typically in
            'YYYY-MM-DD' format.

        Returns
        -------
        DataFrame
            Spark DataFrame resulting from the executed query.
        """
        formatted_query = self.query.format(date=date)
        return self.spark.sql(formatted_query)

    def save(self, df: DataFrame, date: str) -> None:
        """
        Persist the given DataFrame into a Delta table, replacing the slice
        for the specified reference date.

        The table is partitioned by `dtYear`, and the `replaceWhere` option
        ensures that only rows with `dtRef = <date>` are overwritten.

        Parameters
        ----------
        df : DataFrame
            Spark DataFrame to be written.
        date : str
            Reference date used in the `replaceWhere` clause, typically
            matching the `dtRef` column in the DataFrame.
        """
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("replaceWhere", f"dtRef = '{date}'")
            .partitionBy("dtYear")
            .save(f"data/silver/{self.table}")
        )

    def exec(self, iters, compact: bool = False) -> None:
        """
        Run the ingestion pipeline for multiple reference dates.

        Parameters
        ----------
        iters : Iterable[str]
            Iterable of reference dates (e.g., list of 'YYYY-MM-DD' strings).
        compact : bool, optional
            If True, rewrites the entire table (coalesce(1)) and runs VACUUM
            after processing all dates. Default is False to avoid heavy jobs.
        """
        for ref_date in track(iters, description="Running ingestion..."):
            df = self.load(ref_date)
            self.save(df, ref_date)

        if not compact:
            return

        (
            self.spark.read.format("delta")
            .load(f"data/silver/{self.table}")
            .coalesce(1)
            .write.format("delta")
            .mode("overwrite")
            .partitionBy("dtYear")
            .save(f"data/silver/{self.table}")
        )

        delta_table = DeltaTable.forPath(self.spark, f"data/silver/{self.table}")
        delta_table.vacuum()
