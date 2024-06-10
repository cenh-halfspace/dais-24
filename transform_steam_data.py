# Databricks notebook source
# MAGIC %md 
# MAGIC ### Load and Clean Steam Store Data
# MAGIC The dataset we use, is based on a Kaggle open dataset, named Steam Store Data:  
# MAGIC https://www.kaggle.com/datasets/amanbarthwal/steam-store-data?resource=download  
# MAGIC   
# MAGIC The dataset contains about 42.500 steam store video game listings, of which we clean the data a bit, and filter down to 3000 popular and decently reviewed games. This filtering is done, since vector search indexing is a slow process, and we wanted it to be faster. Ideally, we would have used more of the dataset.  
# MAGIC   
# MAGIC We keep the original larger version of the dataset around, but for the Generate notebook, we only use the smaller subset.

# COMMAND ----------

# MAGIC %sql
# MAGIC USE workspace.steam

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Column
from pyspark.sql.types import IntegerType


def transform_review_enum_to_score(target_col: str) -> Column:
    return (
        F.when(
            F.col(target_col) == F.lit("Overwhelmingly Positive"),
            F.lit(9.5),
        )
        .when(F.col(target_col) == F.lit("Very Positive"), F.lit(8.5))
        .when(F.col(target_col) == F.lit("Positive"), F.lit(8.0))
        .when(F.col(target_col) == F.lit("Mostly Positive"), F.lit(7.0))
        .when(F.col(target_col) == F.lit("Mixed"), F.lit(5.0))
        .when(F.col(target_col) == F.lit("Mostly Negative"), F.lit(4.0))
        .when(F.col(target_col) == F.lit("Negative"), F.lit(3.0))
        .when(F.col(target_col) == F.lit("Very Negative"), F.lit(2.0))
        .when(
            F.col(target_col) == F.lit("Overwhelmingly Negative"),
            F.lit(1.0),
        )
        .otherwise(F.lit(None))
    )


steam_games = spark.read.table("steam_games_input")

steam_games_transformed = (
    steam_games.withColumn(
        "overall_review_score", transform_review_enum_to_score("overall_review")
    )
    .withColumn(
        "recent_review",
        F.when(F.col("recent_review").isNull(), F.col("overall_review")).otherwise(
            F.col("recent_review")
        ),
    )
    .withColumn(
        "recent_review_count",
        F.when(F.col("recent_review_count").isNull(), F.lit(0)).otherwise(
            F.col("recent_review_count")
        ),
    )
    .withColumn("recent_review_score", transform_review_enum_to_score("recent_review"))
    .withColumn(
        "IsSelfPublished",
        (F.col("developer") == F.col("publisher"))
        | (F.col("developer").isNotNull() & F.col("publisher").isNotNull()),
    )
    .withColumn("release_year", F.substring("release_date", -4, 5))
    .filter(
        "release_year not in ('True', 'alse', 'nced', 'soon') and release_year is not null"
    )
    .filter("overall_review_count is not null")
    .drop(
        "original_price",
        "discount_percentage",
        "discounted_price",
        "age_rating",
        "content_descriptor",
        "release_year",
    )
)

# COMMAND ----------

display(steam_games_transformed.orderBy(F.rand()))

# COMMAND ----------

print(
    f"Filtered away {steam_games.count() - steam_games_transformed.count()} of {steam_games.count()} rows"
)

# COMMAND ----------

steam_games_transformed.write.format("delta").mode("overwrite").saveAsTable(
    "steam_games_cleaned"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE
# MAGIC   `steam_games_cleaned`
# MAGIC SET
# MAGIC   TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Cut down the size, for Vector Search Indexing

# COMMAND ----------

# Cut down on the size of the table, by only taking properly reviewed and positively rated games into account.
steam_games_filtered = steam_games_transformed.drop(
    "linux_support",
    "mac_support",
    "win_support",
    "recent_review",
    "recent_review_count",
    "recent_review_score",
).filter("overall_review_count > 1000 and overall_review_score >= 8")

print(
    f"Filtered away {steam_games.count() - steam_games_filtered.count()} of {steam_games.count()} rows"
)

steam_games_filtered.write.format("delta").mode("overwrite").saveAsTable(
    "steam_games_smallest"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE
# MAGIC   `steam_games_smallest`
# MAGIC SET
# MAGIC   TBLPROPERTIES (delta.enableChangeDataFeed = true)
