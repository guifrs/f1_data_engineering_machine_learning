# %%
import pyspark
from delta import configure_spark_with_delta_pip

builder = (
    pyspark.sql.SparkSession.builder.appName("MyApp")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
df = spark.read.csv(r"../data/raw/*.csv", sep=";", header=True)

(
    df.coalesce(1)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save("../data/bronze/results")
)

# %%
