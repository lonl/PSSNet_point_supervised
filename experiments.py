def get_experiment(exp_name):
    if exp_name == "trancos":
        dataset_name="trancos"
        model_name="ResFCN"
        metric_name = "MAE"

    if exp_name == "shanghai":
        dataset_name="shanghai"
        model_name="FCN8"
        metric_name = "MAE"

    if exp_name == "pascal":
        dataset_name="pascal"
        model_name="ResFCN"
        metric_name = "mRMSE"
    #
    # if exp_name == "acacia":
    #     dataset_name="acacia"
    #     model_name="ResFCN"
    #     metric_name = "MAE"

    if exp_name == "acacia":
        dataset_name = "acacia"
        model_name = "ResUnet"
        metric_name = "MAE"

    # if exp_name == "acacia":
    #     dataset_name = "acacia"
    #     model_name = "ResUnet"
    #     metric_name = "mRMSE"

    if exp_name == "oilpalm":
        dataset_name = "oilpalm"
        model_name = "ResUnet"
        metric_name = "MAE"

    if exp_name == "Sorghum":
        dataset_name = "Sorghum"
        model_name = "ResUnet"
        metric_name = "MAE"

    print("Model: {} - Dataset: {} - Metric: {}".format(model_name, dataset_name,metric_name))
    return dataset_name, model_name, metric_name
