# ai_results_viz
Scripts for model evaluation and archivation of false prediction

## Dependencies

The pipeline was tested with Python 3.8.10.
Use requirements.txt to setup the python packages.

## Creating ROC curves and confusion matrices

Run the ````plot_results_binary.py```` script to create ROC curves and confusion matrices. The script needs a ````test_scores.csv```` file as input which will be created by ````run_test.sh```` (see *ai_pipeline* repository) and config.json which contains all relevant parameters and the path to ````test_scores.csv````. A new folder (= *1_statistics*) will be created in which the graphs will be saved.

### Input
All relevant paths and parameters need to be filled out in ````config.json````:
````json
{
    "path_to_scores": ".\\dummy_data\\model\\test_scores.csv",
    "classes": ["good", "bad"],
    "class_dict": {"good":0, "bad":1},
    "negative_class": "bad",
    "threshold": 0.5,
    "sample_dict":
    {
        "good":
        [
            ["Good", 0],
            ["Good", 1],
            ["Good", 2]
        ],
        "bad" :
        [
            ["Bad", 0],
            ["Bad", 1],
            ["Bad", 2]
        ]
    }
}
````
| Parameter            |  Description                  |
|----------------------|-------------------------------|
|````path_to_scores````| Path to test_scores.csv|
|````classes````| List of classes|
|````class_dict````| Dictionary mapping each class to its respective prediction value|
|````negative_class````| Negative class of ````classes````|
|````threshold````| Picked threshold for the creation of confusion matrix|
|````sample_dict```` | Dictionary that lists all samples of respective class. See example above (class = "good", samples of "good" class = \[\["Good", 0], \["Bad", 2]] &#8594; {"good": [["Good", 0], ["Bad", 2]} <br> Each sample must have the following format: list("sample_name", "sample_id"). Images must have the following name format: "sample_name"\_"sample_ID"\_"rotation"_"\_"trigger".file_extention|


