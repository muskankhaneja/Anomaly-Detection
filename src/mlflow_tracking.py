import mlflow

def log_results_with_mlflow(X, Y, trainX, cv_results, best_model, cv_scores,
                            test_path, top5_df, **kwargs):
    # Log the results with MLflow
    for i in range(top_n):
        with mlflow.start_run(description=kwargs['description']) as run:
            # Add tag on run
            mlflow.set_tags(tags=kwargs['tags'])

            # Store parameters
            mlflow.log_params(dict(top5_df['params'][i]))

            # Store metrics
            mlflow.log_metrics(top5_df[metrics_columns].iloc[i, :].to_dict())

            if i == 0:
                # Log and save the best model. Include signature
                signature = infer_signature(trainX, best_model.predict(trainX))
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    signature=signature,
                )

                # Make cross validated predictions and plot the results
                predictions = mp.make_cv_predictions(best_model, X, Y)
                vis.plot_confusion_matrix(
                      Y,
                      predictions,
                      log_to_mlflow=True,
                      title="Confusion Matrix on Validation Set",
                )
                vis.plot_cv_scores(cv_scores, log_to_mlflow=True)

                # Make predictions on test data and log metrics
                test_data = log_test_metrics_to_mlflow(
                      best_model,
                     test_path,
                     kwargs['predictions_path'],
                )

                # Log ROC AUC curve
                vis.plot_ROC_AUC_curve(
                    best_model,
                    test_data[test_data.columns.difference(['class'])],
                    test_data['class'],
                    log_to_mlflow=True,
                    title="ROC AUC Curve on Test Data",
                )