import os
import modal
    
BACKFILL=False
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def get_random_passenger(fs):
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    # randomly pick one of these 3 and write it to the featurestore
    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()

    pick_random = random.uniform(0,df.shape[0]-1)

    titanic_df = df.iloc[[pick_random]]

    return titanic_df



def g():
    import hopsworks
    import pandas as pd
    project = hopsworks.login()
    fs = project.get_feature_store()


    if BACKFILL == True:
        titanic_df = pd.read_csv("./titanic_fixed.csv")
    else:
        titanic_df = get_random_passenger(fs)

    columns = list(titanic_df.columns)

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=columns,
        description="Titanic survivors dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
