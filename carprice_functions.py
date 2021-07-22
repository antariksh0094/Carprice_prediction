def fit_predict(estimator):
    estimator.fit(x_train, train_targets)
    train_preds = estimator.predict(x_train)
    val_preds = estimator.predict(x_val)

    from sklearn.metrics import mean_squared_error
    
    train_error = np.sqrt(mean_squared_error(train_targets, train_preds))
    val_error = np.sqrt(mean_squared_error(val_targets, val_preds))
    
    
    print('Train RMSLE:', train_error)
    print('Val RMSLE:', val_error)
    
    # LOGGING MODELLING PARAMS
    
    log = pd.read_excel('model_params_metrics.xlsx')
    
    size= len(log.index)
    log.loc[size] = np.nan
    log.loc[size, 'Date'] = datetime.now()
    log.loc[size,'estimator'] = str(estimator)
    log.loc[size,'hyperparams'] = str(estimator.get_params())[1:-1]
    log.loc[size, 'train_RMSLE'] = train_error
    log.loc[size, 'val_RMSLE'] = val_error
    
    log.to_excel('model_params_metrics.xlsx', index=False)
    log.tail()
    
# def log_model_metrics(estimator):
#     dflog = dflogfinal.values.flatten()
#     dflog = np.append(dflog, [time, str(estimator), params, train_error, val_error])
#     dflogfinal = dflog.reshape(-1, 5)
#     print(pd.DataFrame(dflogfinal, columns = ['time', 'model', 'hyperparams','train_error','val_error']))