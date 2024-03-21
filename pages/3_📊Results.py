import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import t
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import plotly.express as px
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
# import pygwalker as pyg


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://kmkconsultinginc.com/wp-content/uploads/2020/12/KMK-Logo.png);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Menu";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()
# Main header text
main_header_text = "Forecast Results"
st.markdown(f"<h1 style='text-align: center;color: #E82373;'>{main_header_text}</h1>", unsafe_allow_html=True)
st.write('---')
if 'uploaded_file' not in ss:
    ss.uploaded_file=None
if ss.uploaded_file is not None and ss.file_uploaded is not None:
    def calculate_decalendarized_sales(actual_sales, calendarization_values):
        return actual_sales / calendarization_values
    # Function to calculate calendarized sales
    def calculate_calendarized_sales(actual_sales, calendarization_values):
        return actual_sales * calendarization_values
    def convert_to_tuple(input_string):
                # Split the input string by comma and strip any leading or trailing whitespace
                parts = input_string.split(',')
                parts = [part.strip() for part in parts]

                # Convert the parts to integers
                integers = [int(part) for part in parts]

                # Return the tuple
                return tuple(integers)
    
    ##Linear Forecast
    def linear_forecast_decalendarized(data, start_date, end_date, product_column, degree=1):
        # Ensure 'DataPeriodNumber' and the product column are numeric
        data['Dataperiod'] = pd.to_datetime(data['Dataperiod'])
        data['DataPeriodNumber'] = (data['Dataperiod'] - data['Dataperiod'].min()).dt.days
        
        X = data['DataPeriodNumber']
        y = data[product_column]
        
        # Fit a polynomial of specified degree
        coeffs = np.polyfit(X, y, degree)
        slope, intercept = coeffs
        
        # Determine the frequency of the data
        data_period_diff = data['Dataperiod'].diff().dt.days.mean()
        if data_period_diff > 28:  # Assuming monthly data if average difference > 28 days
            freq = 'M'
        else:
            most_common_day = pd.to_datetime(data['Dataperiod']).dt.day_name().mode()[0]
            freq = f'W-{most_common_day[:3].upper()}'  # Construct weekly frequency string
        
        # Generate dates between start_date and end_date with the determined frequency
        future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        future_dataperiods = pd.DataFrame({'DataPeriodNumber': range(data['DataPeriodNumber'].max() + 1, data['DataPeriodNumber'].max() + 1 + len(future_dates))})
        future_predictions = slope * future_dataperiods['DataPeriodNumber'] + intercept

        return pd.DataFrame({'Dataperiod': future_dates,
                            'Linear Forecast': future_predictions})
    
    # def get_previous_13(group):
    #             return group.iloc[13:26] 
    #Exponential Forecast
    def exponential_forecast_decalendarized(data, start_date, end_date, product_column, alpha, seasonal_periods=4, trend='mul', seasonal='add'):
        X = data[['DataPeriodNumber']]
        y = data[product_column]
        
        # Fit exponential smoothing model
        model = ExponentialSmoothing(y,trend=trend, seasonal_periods=seasonal_periods,seasonal=seasonal)
        fitted_model = model.fit()
        
        # Determine the frequency of the data
        data_period_diff = data['Dataperiod'].diff().dt.days.mean()
        if data_period_diff > 28:  # Assuming monthly data if average difference > 28 days
            freq = 'M'
        else:
            most_common_day = pd.to_datetime(data['Dataperiod']).dt.day_name().mode()[0]
            freq = f'W-{most_common_day[:3].upper()}'  # Construct weekly frequency string
        
        # Generate dates between start_date and end_date with the determined frequency
        future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        future_dataperiods = pd.DataFrame({'DataPeriodNumber': range(data['DataPeriodNumber'].max() + 1, data['DataPeriodNumber'].max() + 1 + len(future_dates))})
        future_predictions_exponential = fitted_model.forecast(len(future_dates))
        
        # Calculate standard deviation of errors
        residuals = y - fitted_model.fittedvalues
        std_error = np.std(residuals, ddof=1)  # Sample standard deviation
        
        # Calculate degrees of freedom
        n = len(y)
        if trend is not None:
            p = 2 if seasonal is not None else 1
        else:
            p = 1 if seasonal is not None else 0
        df = n - p
        
        # Calculate t-statistic
        t_stat = t.ppf(1 - alpha / 2, df)
        
        # Calculate confidence intervals
        lower_bound = future_predictions_exponential - t_stat * std_error * np.sqrt(1 + 1 / n)
        upper_bound = future_predictions_exponential + t_stat * std_error * np.sqrt(1 + 1 / n)
        
        # Combine forecast and confidence intervals into a DataFrame
        forecast_df = pd.DataFrame({'Dataperiod': future_dates,
                                    'Exponential Forecast': future_predictions_exponential,
                                    'Lower CI': lower_bound,
                                    'Upper CI': upper_bound})
        
        return forecast_df

    ##Prophet Forecast
    def prophet_forecast_decalendarized(data, start_date, end_date, product_column):
        # Prepare data for Prophet
        data_prophet = data.rename(columns={'Dataperiod': 'ds', product_column: 'y'})
        
        # Initialize Prophet model
        model = Prophet()
        
        # Fit the model
        model.fit(data_prophet)
        
        # Determine the frequency of the data
        data_period_diff = data['Dataperiod'].diff().dt.days.mean()
        if data_period_diff > 28:  # Assuming monthly data if average difference > 28 days
            freq = 'M'
        else:
            most_common_day = pd.to_datetime(data['Dataperiod']).dt.day_name().mode()[0]
            freq = f'W-{most_common_day[:3].upper()}'  # Construct weekly frequency string
        
        # Generate dates between start_date and end_date with the determined frequency
        future_dates = pd.date_range(start=start_date, end=end_date, freq=freq) # Daily frequency
        
        # Create DataFrame for future predictions
        future_dataperiods = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        future_predictions_prophet = model.predict(future_dataperiods)
        
        # Extract relevant columns
        forecast = future_predictions_prophet[['ds', 'yhat']]
        forecast.columns = ['Dataperiod', 'Prophet Forecast']
        
        return forecast
    
    #SARIMA FORECAST
    def sarima_forecast_decalendarized(data, start_date, end_date, product_column, order=(1, 1, 1), seasonal_order=(1, 1, 1,13)):
        # Ensure 'Dataperiod' is in datetime format
        data['Dataperiod'] = pd.to_datetime(data['Dataperiod'])
        
        # Extract relevant columns
        X = data['Dataperiod']
        y = data[product_column]
        
        # Fit SARIMA model
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
        results = model.fit()
        
        # Generate future dates
        data_period_diff = data['Dataperiod'].diff().dt.days.mean()
        if data_period_diff > 28:
            freq = 'M'
        else:
            most_common_day = pd.to_datetime(data['Dataperiod']).dt.day_name().mode()[0]
            freq = f'W-{most_common_day[:3].upper()}'
        
        future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Forecast
        forecast = results.get_forecast(steps=len(future_dates))
        forecast_values = forecast.predicted_mean
        
        # Create DataFrame for forecasted values
        forecast_df = pd.DataFrame({
            'Dataperiod': future_dates,
            'SARIMA Forecast': forecast_values
        })
        
        return forecast_df
    
    #Gaussian Forecast
    def gaussian_forecast_decalendarized(data, start_date, end_date, product_column, kernel=None):
        # Ensure 'Dataperiod' is in datetime format
        data['Dataperiod'] = pd.to_datetime(data['Dataperiod'])
        
        # Extract relevant columns
        X = np.atleast_2d(data['Dataperiod'].astype('int64')).T
        y = data[product_column]
        
        # Define kernel if not provided
        if kernel is None:
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e10))
        
        # Fit Gaussian Process model
        model = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=10)
        model.fit(X, y)
        
        # Determine frequency for future dates
        data_period_diff = data['Dataperiod'].diff().dt.days.mean()
        if data_period_diff > 28:
            freq = 'M'
        else:
            most_common_day = pd.to_datetime(data['Dataperiod']).dt.day_name().mode()[0]
            freq = f'W-{most_common_day[:3].upper()}'
        
        # Generate future dates
        future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Forecast
        forecast_values, _ = model.predict(np.atleast_2d(future_dates.astype(int)).T, return_std=True)
        
        # Create DataFrame for forecasted values
        forecast_df = pd.DataFrame({
            'Dataperiod': future_dates,
            'Gaussian Forecast': forecast_values
        })
        
        return forecast_df
    
    if ss['selected_models'] is not None:        
        
        # Define a function to display an upload icon
        # Function to calculate decalendarized sales
        df1 = ss.uploaded_file
        df2 = ss.uploaded_file1
        df3 = pd.read_excel('Calendarisation.xlsx',sheet_name=1)
        dateparams = ss.parameters
        model = ss['selected_models']
        modelparams = ss['selected_model_params']
        merged_data=pd.merge(df1,df2,on='Dataperiod')             
        # Sort DataFrame by 'Dataperiod' column in descending order
        merged_data = merged_data.sort_values(by='Dataperiod', ascending=False)
        # Add 'DataPeriodNumber' column with values assigned in reverse order
        merged_data['DataPeriodNumber'] = range(1, len(merged_data) + 1)
        merged_data = merged_data[['DataPeriodNumber'] + [col for col in merged_data.columns if col != 'DataPeriodNumber']]
        merged_data=merged_data.sort_values(by='Dataperiod',ascending=True)
        forecast_results = pd.DataFrame()
        dateparams['Baseline Start Date']=pd.to_datetime(dateparams['Baseline Start Date'], format='%Y-%m-%d')
        dateparams['Baseline End Date']=pd.to_datetime(dateparams['Baseline End Date'], format='%Y-%m-%d')
        dateparams['Forecast Start Date']=pd.to_datetime(dateparams['Forecast Start Date'], format='%Y-%m-%d')
        dateparams['Forecast End Date']=pd.to_datetime(dateparams['Forecast End Date'], format='%Y-%m-%d')
        

        for product_column in merged_data.columns[2:]:
            product_name = product_column.replace(' sales', '')
            # Empty the selected_data DataFrame
            selected_data = pd.DataFrame()
            
            # Select data from period 01 to baseline number            
            selected_data = merged_data[(merged_data['Dataperiod'] <= dateparams['Baseline End Date']) & (merged_data['Dataperiod'] >= dateparams['Baseline Start Date'])]


            # Calculate decalendarized sales
            selected_data['DecalendarizedSales'] = calculate_decalendarized_sales(selected_data[product_column], selected_data['Calenderization'])
            selected_data['DataPeriodNumber'] = pd.to_numeric(selected_data['DataPeriodNumber'], errors='coerce')
            selected_data['DecalendarizedSales'] = pd.to_numeric(selected_data['DecalendarizedSales'], errors='coerce')
            # Calculate linear forecast on decalendarized sales
            linear_forecast_result = linear_forecast_decalendarized(selected_data, dateparams['Forecast Start Date'], dateparams['Forecast End Date'], 'DecalendarizedSales')
            lf=linear_forecast_result.merge(df2,on='Dataperiod')
            lf['CalendarizedSales']=calculate_calendarized_sales(lf['Linear Forecast'], df2['Calenderization'])

            if 'Linear' in model:                    
                # Store results in the forecast_results DataFrame
                forecast_results[f"{product_name} Linear Forecast"] = lf['CalendarizedSales'].round(1)

            if 'Prophet' in model:
                prophet_forecast_result = prophet_forecast_decalendarized(selected_data, dateparams['Forecast Start Date'], dateparams['Forecast End Date'], 'DecalendarizedSales')
                pf= prophet_forecast_result.merge(df2,on='Dataperiod')
                pf['CalendarizedSales']=calculate_calendarized_sales(pf['Prophet Forecast'], df2['Calenderization'])
                forecast_results[f"{product_name} Prophet Forecast"] = pf['CalendarizedSales'].round(1)

            if 'Sarima' in model:    
                sarima_forecast_result = sarima_forecast_decalendarized(selected_data, dateparams['Forecast Start Date'], dateparams['Forecast End Date'], 'DecalendarizedSales',ss['model_params']['Sarima'][0] ,ss['model_params']['Sarima'][1] )
                sf=sarima_forecast_result.merge(df2,on='Dataperiod')
                sf['CalendarizedSales']=calculate_calendarized_sales(sf['SARIMA Forecast'], df2['Calenderization'])
                forecast_results[f"{product_name} SARIMA Forecast"] = sf['CalendarizedSales'].round(1)

            if 'Exponential' in model:
                exponential_forecast_result = exponential_forecast_decalendarized(selected_data,dateparams['Forecast Start Date'], dateparams['Forecast End Date'], 'DecalendarizedSales', 1 - modelparams['Exponential'][3] / 100,seasonal_periods=modelparams['Exponential'][2], trend=modelparams['Exponential'][0], seasonal=modelparams['Exponential'][1])
                ef=exponential_forecast_result.merge(df2,on='Dataperiod')
                ef['CalendarizedSales']=calculate_calendarized_sales(ef['Exponential Forecast'], df2['Calenderization'])
                ef['UpperLimits']=calculate_calendarized_sales(ef['Upper CI'], df2['Calenderization'])
                ef['LowerLimits']=calculate_calendarized_sales(ef['Lower CI'], df2['Calenderization'])
                forecast_results[f"{product_name} Exponential Forecast"] =ef['CalendarizedSales'].round(1)
                forecast_results[f"{product_name} Upper Limit"] =  ef['UpperLimits'].round(1)
                forecast_results[f"{product_name} Lower Limit"] =  ef['LowerLimits'].round(1)
            
            if 'Gaussian' in model:
                gaussian_forecast_result = gaussian_forecast_decalendarized(selected_data, dateparams['Forecast Start Date'], dateparams['Forecast End Date'], 'DecalendarizedSales')
                gf=gaussian_forecast_result.merge(df2,on='Dataperiod')
                gf['CalendarizedSales']=calculate_calendarized_sales(gf['Gaussian Forecast'], df2['Calenderization'])                             
                forecast_results[f"{product_name} Gaussian Forecast"] = gf['CalendarizedSales'].round(1)   
            
        # Include the date column in the forecast_results DataFrame
        forecast_results['Date'] = lf['Dataperiod']

        # Set the index to product and forecast type for better organization
        forecast_results.set_index(['Date'], inplace=True)
        # Transpose the DataFrame
        forecast_results_transposed = forecast_results.T
        df=forecast_results_transposed.reset_index()
        df=df.rename_axis(index=None,columns=None)
        df.rename(columns={'index':'Product'},inplace=True)
        df = df[~df['Product'].str.contains('Calenderization')]
        # Reshaping the DataFrame
        df = pd.melt(df, id_vars=['Product'], var_name='Date', value_name='Value')
        try:
            df[['Product', 'Territory', 'Method','Forecast']] = df['Product'].str.split(' ', expand=True)
            df['Forecast Method']=df['Method']+' '+df['Forecast']
            df.drop(columns=['Method','Forecast'], inplace=True)
            
            df3t=df3[(df3['Dataperiod']>=dateparams['Forecast Start Date'])&(df3['Dataperiod']<=dateparams['Forecast End Date'])]
            # df3t=df3t[df3t['Dataperiod'].dt.month==df3t['Month'].dt.month]
            result = pd.concat([df3t.iloc[[1,-2]]])
            result['month']=result['Month'].dt.month
            result.rename(columns={'Dataperiod':'Date'},inplace=True)
            result=result[['Date','month','Split']]
            dfs=df.copy()
            dfs['Date'] = pd.to_datetime(dfs['Date'])
            dfs['month']=dfs['Date'].dt.month
            finaldf=dfs.merge(result,how='left',on=['Date','month'])
            finaldf['Split'].fillna(1,inplace=True)
            date_column=finaldf['Date'].dt.date
            
            
            finaldf['Value'] = finaldf['Value'].multiply(finaldf['Split'], axis=0)
            final_df = pd.concat([date_column,finaldf.drop(columns=['Date','month','Split'])], axis=1)

            if dateparams['selected_type']=='Quarterly':
                pivot_df = final_df.pivot_table(index=['Product', 'Territory', 'Forecast Method'], columns='Date', values='Value')
                pivot_df['Total'] =  pivot_df.sum(axis=1).round(1)
                dfgraph=final_df.copy()
            else:
                dfp=df.copy()
                dfp['Date'] = pd.to_datetime(df['Date'])
                dfp['Date']=dfp['Date'].dt.date
                pivot_df = dfp.pivot_table(index=['Product', 'Territory', 'Forecast Method'], columns='Date', values='Value')
                pivot_df['Total'] =  pivot_df.sum(axis=1).round(1)
                dfgraph=dfp.copy()
                # Assuming pivot_df is your DataFrame
            total_column = pivot_df.pop('Total')  # Extract 'Total' column
            pivot_df.insert(0, 'Total', total_column)  # Insert 'Total' column at the first position
            st.markdown("<h3 style='color: #E82373;'>Forecasted Data</h3>", unsafe_allow_html=True)                      
            st.dataframe(pivot_df)
            
            dft=ss.uploaded_filed
            dft1=pd.melt(dft,id_vars=['Product','Territory'],var_name='Date',value_name='Value')
            # merged_df = pd.merge(dft1, df, on=['Product', 'Territory', 'Date'], how='outer')
            st.markdown("<h3 style='color: #E82373;'>Forecasted Graphs</h3>", unsafe_allow_html=True)
            st.subheader('Product Forecast')
            # Get unique products and territories
            unique_products = dft1['Product'].unique()
            unique_territories = dft1['Territory'].unique()

            # Multiselect box for selecting products and territories
            d1,d2=st.columns(2)
            selected_products = d1.multiselect('Select Products', unique_products, default=unique_products)
            selected_territories = d2.multiselect('Select Territories', unique_territories, default=unique_territories)
            #connecting baseline with forecast
            max_date_row = dft1.loc[dft1['Date'] == dft1['Date'].max()]
            result_df = pd.DataFrame(columns=['Product', 'Territory', 'Date', 'Value', 'Forecast Method'])
            for model in dfgraph['Forecast Method'].unique():
                temp_df = max_date_row.copy()
                if 'Linear' in model:
                    temp_df['Forecast Method']= 'Linear Forecast'
                if 'Sarima' in model:
                    temp_df['Forecast Method']= 'Sarima Forecast'
                if 'Exponential' in model:
                    temp_df['Forecast Method']= 'Exponential Forecast'
                if 'Upper' in model:
                    temp_df['Forecast Method']= 'Upper Limit'
                if 'Lower' in model:
                    temp_df['Forecast Method']= 'Lower Limit'
                if 'Gaussian' in model:
                    temp_df['Forecast Method']= 'Gaussian Forecast'
                if 'Prophet' in model:
                    temp_df['Forecast Method']= 'Prophet Forecast'
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
            result_df['Date']=pd.to_datetime(result_df['Date'])
            result_df['Date']=result_df['Date'].dt.date
            combined_df = pd.concat([result_df, dfgraph],ignore_index=True)
            
            # Filter data based on selected products and territories
            filtered_dft1 = dft1[(dft1['Product'].isin(selected_products)) & (dft1['Territory'].isin(selected_territories))]
            filtered_df = combined_df[(combined_df['Product'].isin(selected_products)) & (combined_df['Territory'].isin(selected_territories))]
                
            
            # Plot product and territory-wise
            for product in selected_products:
                for territory in selected_territories:
                    # Filter data for the current product and territory
                    dft1_data = filtered_dft1[(filtered_dft1['Product'] == product) & (filtered_dft1['Territory'] == territory)]
                    df_data = filtered_df[(filtered_df['Product'] == product) & (filtered_df['Territory'] == territory)]
                    
                    # Plot with Plotly Express
                    fig = go.Figure()

                    # Add trace for data from the first dataframe
                    fig.add_trace(go.Scatter(x=dft1_data['Date'], y=dft1_data['Value'], mode='lines', name='Baseline'))

                    # Add traces for data from the second dataframe with 'Forecast Method' as legend
                    colors = px.colors.qualitative.Set1  # Get a list of colors from the Plotly palette
                    for i, (method, data) in enumerate(df_data.groupby('Forecast Method')):
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Value'], mode='lines', name=method, line=dict(dash='dash', color=colors[i+2])))

                    # Update layout
                    fig.update_layout(title=f'{product} in {territory}', height=500, width=1350)

                    # Streamlit app
                    st.plotly_chart(fig)
                            


            st.markdown('---')
            st.subheader('Combined Forecast ')
            unique_product = dft1['Product'].unique()
            unique_territorie = dft1['Territory'].unique()
            list_of_forecast_methods = combined_df['Forecast Method'].unique()
            z1, z2,z3 = st.columns(3)
            selected_product = z1.multiselect('Select Products', unique_product, default=unique_product, key='select_products')
            selected_territorie = z2.multiselect('Select Territories', unique_territorie, default=unique_territorie, key='select_territories')
            f3 = z3.multiselect('Select Forecast Model',options=list_of_forecast_methods)
            # Create a new dataframe for aggregated data
            filtered_baseline=filtered_dft1.copy()
            filtered_baseline['Forecast Method']='Baseline'
            chart2_data=pd.concat([filtered_baseline,filtered_df])
            fig = px.line()
            fig.data = []
            traces = []
            

            for prod in selected_product:
                for terr in selected_territorie:
                    # Add baseline trace
                    condition_baseline = (
                        (chart2_data['Product'] == prod) &
                        (chart2_data['Territory'] == terr) &
                        (chart2_data['Forecast Method'] == 'Baseline')
                    )
                    df_baseline = chart2_data[condition_baseline]
                    if not df_baseline.empty:
                        fig.add_trace(go.Scatter(x=df_baseline['Date'], y=df_baseline['Value'], mode='lines', name=f"{prod} - Baseline"))

                    for fm in f3:
                        condition_forecast = (
                            (chart2_data['Product'] == prod) &
                            (chart2_data['Territory'] == terr) &
                            (chart2_data['Forecast Method'] == fm)
                        )

                        df_forecast = chart2_data[condition_forecast]

                        if not df_forecast.empty:
                            fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Value'], mode='lines', name=f"{prod} - {fm}", line=dict(dash='dash')))

            # Update layout
            fig.update_layout(title='Product and Territory-wise', height=500, width=1200)

            # Streamlit app
            st.plotly_chart(fig, use_container_width=True)
            

            st.subheader('Summary')
            # Assuming dateparams['Baseline Start Date'] and dateparams['Baseline End Date'] are Timestamp objects
            st.write(dateparams['selected_type'],'Forecast')
            data = {
                'Parameter': ['Baseline Start Date', 'Baseline End Date', 'Forecast Start Date', 'Forecast End Date'],
                'Date': [
                    dateparams['Baseline Start Date'].date(),
                    dateparams['Baseline End Date'].date(),
                    dateparams['Forecast Start Date'].date(),
                    dateparams['Forecast End Date'].date()
                ]
            }
            df = pd.DataFrame(data)
            s1 = dict(selector='th', props=[('text-align', 'center')])
            s2 = dict(selector='td', props=[('text-align', 'center')])
            # you can include more styling paramteres, check the pandas docs
            table = df.style.set_table_styles([s1,s2]).hide(axis=0).to_html()     
            st.write(table, unsafe_allow_html=True)
            
            f13=combined_df.groupby(['Product','Forecast Method']).apply(lambda x: x[x['Date'] != x['Date'].min()]['Value'].tail(13).sum()).reset_index()
            f13.rename(columns={0:'Value'},inplace=True)
            
            mr13 = filtered_dft1.groupby('Product').tail(13)
            sorted_df = filtered_dft1.sort_values(by=['Product', 'Date'], ascending=[True, False])
            # Group by 'Product' and assign a sequential count within each group
            sorted_df['WeekCount'] = sorted_df.groupby('Product').cumcount()
            # Filter the rows where the count falls within the range of the previous 13 weeks
            p13 = sorted_df[sorted_df['WeekCount'].between(13, 25)]
            # Drop the WeekCount column if not needed
            p13.drop(columns=['WeekCount'], inplace=True)
            # f13=combined_df.head(26)         
            product_mr13 = mr13.groupby('Product')['Value'].sum().round(1)
            
            
            product_p13 = p13.groupby('Product')['Value'].sum().round(1)
            # product_f13=f13.groupby(['Product','Forecast Method'])['Value'].sum()
            # Combine product_mr13 and product_p13 into a single DataFrame
            combined_df1 = pd.concat([product_mr13, product_p13], axis=1)

            # Set column names
            combined_df1.columns = ['Most Recent 13 Weeks Sales', 'Previous 13 Weeks Sales']
            
            # Reset index to make 'Product' a column
            combined_df1.reset_index(inplace=True)

            # Set 'Product' as the index
            combined_df1.set_index('Product', inplace=True)
            f13_p = f13.pivot(index='Forecast Method', columns='Product', values='Value')
            combined_f=combined_df1.T
            combined_f.reset_index(inplace=True)
            combined_f.rename(columns={'index':'Forecast Method'},inplace=True)
            f13_p.reset_index(inplace=True)
            combined_df12=pd.concat([combined_f,f13_p])
            combined_df12.reset_index(inplace=True,drop=True)
            df = pd.DataFrame(combined_df12)

            # Calculate growth percentage
            most_recent_sales = df.loc[df['Forecast Method'] == 'Most Recent 13 Weeks Sales']
            previous_sales = df.loc[df['Forecast Method'] == 'Previous 13 Weeks Sales']

            # Calculate growth percentage for each column
            for col in df.columns[1:]:
                df[col + ' Current 13 weeks Growth %'] = ((df[col] - most_recent_sales[col].iloc[0]) / most_recent_sales[col].iloc[0]) * 100
                df[col + ' Current 13 weeks Growth %']=df[col + ' Current 13 weeks Growth %'].round(1).map('{:.2f}%'.format)

            df_rounded=df
             # Replace "0.00%" with "-"
            # df_rounded = df_rounded.replace('0%', '%', regex=True)
            # df_rounded = df_rounded.astype(str).replace('\0.0', '', regex=True)
            # df_rounded = df_rounded.replace('0.0%', '-', regex=True)
            df_rounded = df.round(1).astype(str).replace({'0.00%': '-', '\.0%': '%'}, regex=True)

            s1 = dict(selector='th', props=[('text-align', 'center')])
            s2 = dict(selector='td', props=[('text-align', 'center')])
            # you can include more styling paramteres, check the pandas docs
            table = df_rounded.style.set_table_styles([s1,s2]).hide(axis=0).to_html()     
            st.write(table, unsafe_allow_html=True)
                      

            
            
           
        except ValueError:
           
        #     # Catch the ValueError and display a message asking the user to select a model first
             st.warning('Please select the parameters', icon="🚨")
    

            
else:
    st.warning("⚠️ Please upload a file to see the results")
        
        
        























   
