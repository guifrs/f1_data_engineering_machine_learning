from pathlib import Path
import pyspark
from delta import configure_spark_with_delta_pip
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_view_from_path(spark, relative_path: str):
    path = (PROJECT_ROOT / relative_path).as_posix()
    df = spark.read.format("delta").load(path)
    view_name = Path(relative_path).name
    df.createOrReplaceTempView(view_name)


def read_query(table: str) -> str:
    path = PROJECT_ROOT / "sql" / f"{table}.sql"
    return path.read_text(encoding="utf-8")


def create_table_from_query(spark, table: str):
    query = read_query(table)
    df = spark.sql(query)
    out_path = (PROJECT_ROOT / "data" / "silver" / table).as_posix()
    (
        df.coalesce(1)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(out_path)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create silver Delta tables from SQL queries."
    )
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="Table name (without .sql).",
    )
    args = parser.parse_args()

    builder = (
        pyspark.sql.SparkSession.builder.appName("MyApp")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    create_view_from_path(spark, "data/bronze/results")
    create_table_from_query(spark, args.table)


if __name__ == "__main__":
    main()
