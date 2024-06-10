# Databricks notebook source
# MAGIC %md
# MAGIC ### Sample input:
# MAGIC - I want a fantasy game with spanish influences, playing as a gnome, where I save the world from great evil. Preferably it also has unicorns in it. It should also be dark
# MAGIC - Give me a game based on the Data + AI Summit 2024 from Databricks experience. Make it exciting, and based on Spark.
# MAGIC - Hard Sci-fi game, difficult, with space combat and a wacky side-kick. Let me explore as far as the eye can see, give me interesting NPCs with deep heartfelt stories, that give me a satisfying ending leaving me sad that it is over.

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Install and setup

# COMMAND ----------

!pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Configuration and input

# COMMAND ----------

dbutils.widgets.text("user_input", "I want to play a fantasy game that includes genres like RPG and world-building")

# COMMAND ----------

# Input
text_input_from_user = dbutils.widgets.get("user_input")

# Summarize
max_words_to_use = 10

# Vector search
index_name = "workspace.steam.description_embeddings"
endpoint_name = "testing"
columns_to_use = ["title", "categories", "genres", "about_description"]

# Configuration
model = "databricks-meta-llama-3-70b-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Summarize user input based into few key words using AI Functions

# COMMAND ----------

query = f"SELECT ai_query(\'{model}\', \'Based on a game pitch create {max_words_to_use} themes, no more than {max_words_to_use} themes are allowed. Do not use any words from the text. Only respond with the chosen words seperated by commas. Do not say Here are the 10 keywords. The text is: {text_input_from_user}\')"
print(query)

display(spark.sql(query))

# COMMAND ----------

resulting_keywords = spark.sql(query).collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Find similar games
# MAGIC The current index, description_embeddings, is based off of a filtered selection of games.  
# MAGIC We only take games with more than 1000 reviews, and which are positively rated.  
# MAGIC This limitation is from vector search indexing being relatively slow, and we are working under time :) 

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()
 
index = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
 
results = index.similarity_search(
    query_text=resulting_keywords,
    columns=columns_to_use,
    num_results=3,
    query_type="hybrid"
    )

# COMMAND ----------

results

# COMMAND ----------

game_descriptions = ""
i = 1
for game in results['result']['data_array']:
    game_descriptions += f"Game #{i} has title: {game[0]} and has the following categories: {game[1]}, and the following genres: {game[2]} and the description is {game[3]}"
    i += 1
print(game_descriptions)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Generate new game based on similar games using AI Functions

# COMMAND ----------

# Santize input
game_descriptions = game_descriptions.replace("'", "")

# COMMAND ----------

# Query and present results
generative_query = f"SELECT ai_gen('Come up with a new and unique game, taking inspiration in the following games. The game that you come up with cannot be the same. Do not use any of the same words in the title, and make a description that is distinct from the inspiration. You should provide a title, genres and a description of the game. Your inspiration is: {game_descriptions}')"
display(spark.sql(generative_query))

# COMMAND ----------

# Query and present results
generative_query = f"SELECT ai_gen('Come up with a single new and unique game, based on the following original prompt: {text_input_from_user}. You should generate a single game idea, with a title, genre list and a description of the game. Take creative inspiration from the following existing game descriptions, from which you are not allowed to use the same words as the titles, and your game should be notably distinct from the inspiration. Your inspiration is: {game_descriptions}')"  

display(spark.sql(generative_query))

# COMMAND ----------


