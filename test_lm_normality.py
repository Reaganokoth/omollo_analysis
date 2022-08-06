def test_linear_regression_assumptions(data, predictor, response, control):
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.compat import lzip
    import statsmodels.stats.api as sms
    from statsmodels.stats.diagnostic import het_white
    


    from patsy import dmatrices

    #import statsmodels.api as sm

    #%matplotlib inline


    #long_data_as_npArray = long_data.to_numpy()

    #response_var_list = ['ROA', 'ROE']
    #predictor_var_list = ['EM', 'RU', 'EI']

    #control = 'FS'


    #fig, scatter = plt.subplots(figsize=(11.85,5.5))
    #fig, residual_independece = plt.subplots(figsize=(11.85,5.5))
    #fig, residual_normality = plt.subplots(figsize=(11.85,5.5))



    ##############################
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """





    #############################
    for response_var in response:
        print (f'----------------------------------TESTING FOR ASSUMPTIONS FOR {response_var}-----------------------------------')

        print('\n============================================================================================')

        print('Assumption 1: Linear Relationship between the PREDICTOR and the RESPONSE variables')
        print('\nAssumption 2: RESIDUAL ERRORS are: random,independent')

        print('\nAssumption 3: Residual  errors are normally distributed')

        print('\nAssumption 4: Residual errors are homoscedastics')

        print('===============================================================================================')


        for predictor_var in predictor:
            plt.figure(figsize=(15, 10),dpi = 100)


            sns.lmplot(x=predictor_var, y=response_var,data = data, fit_reg=True, palette='viridis', height=5, aspect=2)
            plt.ylabel(f'{response_var}',fontsize=15)
            plt.xlabel(f'{predictor_var}',fontsize=15)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.title(f'Assumption 1: Linearity \n{response_var} vs {predictor_var}', fontsize=15)
            #plt.show()


            '''
            Residual Error: Assumes that residual errors left over
                            from fitting the model to the data are independent,
                            identically distributed random variables

            '''

            model_expr = f'{response_var} ~ {predictor_var} + {control}'
            y, X = dmatrices(model_expr, data, return_type='dataframe')
            mask = np.random.rand(len(X)) < 0.8
            X_train = X[mask]
            y_train = y[mask]
            X_test = X[~mask]
            y_test = y[~mask]

            olsr_results = sm.OLS(y_train, X_train).fit()
            olsr_predictions = olsr_results.get_prediction(X_test)
            prediction_summary_frame = olsr_predictions.summary_frame()

            #print('Training completed')

            #print(olsr_results.summary())
            print('Training completed')

            resid = y_test[response_var] - prediction_summary_frame['mean']

            fig, ax = plt.subplots(figsize=(11.85,5.5))
            ax.scatter(y_test[response_var], resid, s=10, c='lime', alpha = 0.7, marker ='o')

            ax.set_xlabel(f'Predicted {response_var}', fontsize=18)
            ax.set_ylabel('Residual Error of Regression', fontsize=18)
            ax.set_title(f'Assumprion 2 :Residual  errors are independent and normal \n Model: {model_expr}', fontsize=18)

            #plt.show()



            '''
            Assumpption 3: The residual errors should all have a normal distribution with a mean of zero.
                           In statistical language:
            '''


            fig, ay = plt.subplots(figsize=(11.85,5.5))

            ay.hist(resid)
            #sns.distplot(resid)

            ay.set_xlabel(f'Predicted {response_var}', fontsize=18)
            ay.set_ylabel('Residual Error of Regression', fontsize=18)
            ay.set_title(f'Assumprion 3 :Residual  errors are normally distributed\n Model: {model_expr}', fontsize=18)

            #resid.hist(bins=50)
            plt.show()
            

            name = ['Jarque-Bera test', 'Chi-squared(2) p-value', 'Skewness', 'Kurtosis']

            #run the Jarque-Bera test for Normality on the residuals vector
            test = sms.jarque_bera(resid)
            
            print('\nPerfoming the  Jarque Bera Test for residual ...')
            
            print('\n==================================================================================================================\n')
            
            print(f'RESIDUAL NORMALITY TEST RESULTS FOR ({model_expr}) MODEL')
            
            print('\n==================================================================================================================')
            
            print(pd.DataFrame({
                'stat':name,
                'value':test
            }))

            
            if test[1] < 0.01:
                print(' **** Residuals of the linear regression model are for all practical purposes not normally distributed ****')
            else:
                print('**** Residuals of the linear regression model are normally distributed ****')

            print('\n==================================================================================================================\n')
            # Assumption 4:
            '''
            Assumpption 4: Residual error homoscedasity:
            '''

            keys = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
            #run the White test
            results = het_white(resid, X_test)

            #print the results. We will get to see the values of two test-statistics and the corresponding p-values
            lzip(keys, results)
            
            print('Perfoming the  White test for heteroscedasticity ...')
            
            print('\n==================================================================================================================\n')
            
            print(f'RESIDUAL HOMOCEDACITY TEST RESULTS FOR ({model_expr}) MODEL')
            
            print('\n==================================================================================================================')
            
            print(pd.DataFrame({
                'stat':keys,
                'value':results
            }))

            
            if results[3] < 0.01:
                print(' **** Residuals of the linear regression model are for all practical purposes not homocedastic ****')
            else:
                print('**** Residuals of the linear regression model are homocedastic ****')

            print('\n==================================================================================================================\n')
        


