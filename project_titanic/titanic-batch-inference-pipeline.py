import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=2)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    # print(y_pred)
    offset = 1

    survived = y_pred[y_pred.size-offset]
    img_urls=["https://uxwing.com/wp-content/themes/uxwing/download/health-sickness-organs/skull-icon.png", "https://uxwing.com/wp-content/themes/uxwing/download/emoji-emoticon/happy-icon.png"]
    expl_string=["They did not survive", "They survived"]

    img_url = img_urls[survived]
    # print(expl_string[survived])

    img = Image.open(requests.get(img_url, stream=True).raw)
    img.save("./latest_titanic.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_titanic.png", "Resources/images", overwrite=True)
    
    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()
    label = df.iloc[-offset]["survived"]
    label_url = img_urls[label]
    print("Actual status: " + expl_string[label])
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_titanic.png")
    dataset_api.upload("./actual_titanic.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survived],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent_titanic.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_titanic.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our titanic_predictions feature group has examples of all 2 survivor status
    print("Number of different survivor status predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Survivor', 'True Casualty'],
                             ['Pred Survivor', 'Pred Casualty'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_titanic.png")
        dataset_api.upload("./confusion_matrix_titanic.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different survivor status predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different survivor status predictions")


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

