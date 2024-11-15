import streamlit as st
from streamlit import session_state as ss
from api.functions import *
from datetime import timedelta
from datetime import datetime
import numpy as np
import pandas as pd

# for models -
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import t
from prophet import Prophet

# for graph-
import plotly.express as px

st.set_page_config(
    page_title="Forecasting",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon='📈'
)

add_logo()

#   #   #   #   #   #   #   #
# Session State Management - 
if 'file_up' not in ss: # for weekly file
    ss.file_up = False
if 'model_in' not in ss: # One Models params are loaded in , called on PROCESS button
    ss.model_in = False

if 'trend_option' not in ss: 
    ss.trend_option = None

if 'seasonality_option' not in ss: 
    ss.seasonality_option = None

if 'seasonal_period' not in ss: 
    ss.seasonal_period = None

if 'confidence_interval' not in ss: 
    ss.confidence_interval = None

if 'kernel_type' not in ss: 
    ss.kernel_type = None

if 'gausian_alpha' not in ss: 
    ss.gausian_alpha = None

if 'length_scale' not in ss: 
    ss.length_scale = None

if 'periodicity' not in ss: 
    ss.periodicity = None

if 'order_input' not in ss: 
    ss.order_input = None

if 'seasonal_order_input' not in ss: 
    ss.seasonal_order_input = None
#   #   #   #   #   #   #   #

#   #   #   #   #   #   #   #
# Global Variables - 
training_points = None
forecast_points = None
prediction_dates = None
weekly = None

#pulling the params from get_model_mods() into global scope-
for var in['trend_option','seasonality_option','seasonal_period','confidence_interval','kernel_type','gausian_alpha','length_scale','periodicity','order_input','seasonal_order_input']:
    globals()[var] = None
#   #   #   #   #   #   #   #

# This function renders the section to find training and testing points - also tempers the weekly df to reduced nobs
def get_ranges():
    global training_points,forecast_points,prediction_dates,weekly
    st.markdown('---')
    c1,c2 = st.columns(2)

    with c1:
        st.subheader('Tweak Training Range : ')
        training_points = st.selectbox(
            label = 'Number of data points used to train forecasting models',
            options = [i for i in range(len(ss['weekly']['date']),6,-1)]
        )
        st.markdown(
            f'''
            <h4>Training Range for Models - 
            <u><b>{ss['weekly'].tail(training_points)['date'].min()}</b></u> to
            <u><b>{ss['weekly']['date'].max()}</b></u></h4>
            '''
            ,unsafe_allow_html=True
        )

    with c2:
        st.subheader('Tweak Forecast Range : ')
        forecast_points = st.selectbox(
            label = 'Number of data points to forecast sales for',
            options = [i for i in range(26,1,-1)]
        )

        # calculating list of future dates - 
        prediction_dates = [ss['weekly']['date'].max() + timedelta(weeks=i) for i in range(1, forecast_points + 1)]

        st.markdown(
            f'''
            <h4>Forecast Range for Models - 
            <u><b>{min(prediction_dates)}</b></u> to
            <u><b>{max(prediction_dates)}</b></u></h4>
            '''
            ,unsafe_allow_html=True
        )
    st.markdown('---')

    # UTIL -
    # Reduction of ss['weekly'] to reduced dataset if needed -
    start = ss['weekly'].tail(training_points)['date'].min()
    end = ss['weekly']['date'].max()
    weekly = ss['weekly'].copy()
    weekly = weekly[(weekly['date'] >= start) & (weekly['date'] <= end)]
    weekly['type'] = 'historical'

# TO DO : MANAGE STORAGE OF PARAMS IN SS | Configure default values
def get_model_mods():

    global trend_option,seasonality_option,seasonal_period,confidence_interval,kernel_type,gausian_alpha,length_scale,periodicity,order_input,seasonal_order_input
    st.markdown("<h2 style='text-align: center;'><u>Model Modifers</u></h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: left;'>1. Linear</h3>", unsafe_allow_html=True)
    with st.expander('About Model'):
        st.markdown(
            f'''
            <p style = '
                border: 2px solid #888888;      
                padding: 10px;
                border-radius: 5px;
                background-color: #2b2b2b;      
                color: #e5e5e5;                
            '>
            This model estimates a <i>linear relationship</i> between the target variable (e.g., sales volume) 
            and time, assuming that sales increase or decrease at a constant rate. 
            It’s straightforward, interpretable, and works well for time series without seasonal or cyclical patterns.<br>
            <u>Trained on</u> <b>{training_points}</b> data points, requires <b>no additional parameters</b>.
            </p>
            ''' , unsafe_allow_html=True
        )

    st.markdown("<h3 style='text-align: left;'>2. Exponential</h3>", unsafe_allow_html=True)
    with st.expander('About Model'):
        st.markdown(
            f'''
            <p style = '
                border: 2px solid #888888;      
                padding: 10px;
                border-radius: 5px;
                background-color: #2b2b2b;      
                color: #e5e5e5;                
            '>
            Exponential Smoothing captures trends by <i>giving more weight to recent observations</i>, making it effective for data with a trend or 
            slight seasonality. It can be adjusted to smooth out fluctuations and is commonly used in short-term forecasting when quick adaptation to recent 
            changes is needed.<br>
            <u>Trained on</u> <b>{training_points}</b> data points, with parameters to control <b>trend and seasonality</b> components.
            </p>
            ''' , unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <ul style = 'border: 2px solid #888888;padding: 10px;border-radius: 5px;background-color: #2b2b2b;color: #e5e5e5;'>
                <li><strong>Trend Options:</strong>
                <ul>
                    <li><strong>'add':</strong> Adds a linear trend component to the model.</li>
                    <li><strong>'mul':</strong> Multiplies the data by an exponentially weighted trend.</li>
                    <li><strong>None:</strong> No trend component is included in the model.</li>
                </ul>
                </li>
                <li><strong>Seasonal Options:</strong>
                <ul>
                    <li><strong>'add':</strong> Adds seasonal variations to the model in an additive manner.</li>
                    <li><strong>'mul':</strong> Incorporates seasonal variations by multiplying the data by seasonal factors.</li>
                    <li><strong>None:</strong> No seasonal component is included in the model.</li>
                </ul>
                </li>
            </ul>
            ''' , unsafe_allow_html=True
        )
    e1,e2,e3,e4 = st.columns(4)
    trend_option = e1.selectbox("Trend", options=['add', 'mul', None],index = 0)
    seasonality_option = e2.selectbox("Seasonality", options=['add', 'mul', None],index = 0)
    seasonal_period = e3.number_input("Seasonal Period (weeks)", min_value=2, max_value=105,value=4)
    confidence_interval = e4.number_input("Confidence Interval (%)", min_value=0.0, max_value=1.0,value = 1.0)

    with st.expander('Explore Other Models '):
        st.markdown("<h3 style='text-align: left;'>3. Gaussian</h3>", unsafe_allow_html=True)
        st.markdown(
            f'''
            <p style = '
                border: 2px solid #888888;      
                padding: 10px;
                border-radius: 5px;
                background-color: #2b2b2b;      
                color: #e5e5e5;                
            '>
            This probabilistic model is useful when sales follow an irregular or complex pattern. 
            By assuming data points come from a <i>Gaussian process</i>, it can model uncertainties and handle non-linear trends well. 
            It may, however, require more computation and fine-tuning of kernel parameters.<br>
            <u>Trained on</u> <b>{training_points}</b> data points, requires <b>specification of kernel functions</b> for smoothness and trend.
            </p>
            ''' , unsafe_allow_html=True
        )
        g1,g2,g3,g4 = st.columns(4)
        kernel_type = g1.selectbox('kernel',options = ['rbf', 'periodic', 'rational_quadratic', 'both'],index = 0)
        gausian_alpha = g2.number_input('alpha',value=1e-3)
        length_scale = g3.number_input('length_scale',value=0.1)
        periodicity = g4.number_input('periodicity',value = 13)

        st.write('')
        st.markdown("<h3 style='text-align: left;'>3. Sarima</h3>", unsafe_allow_html=True)
        st.markdown(
            f'''
            <p style = '
                border: 2px solid #888888;      
                padding: 10px;
                border-radius: 5px;
                background-color: #2b2b2b;      
                color: #e5e5e5;                
            '>
            SARIMA extends ARIMA to include <i>seasonality</i>. It captures trends, cycles, and seasonal patterns, making it suitable for sales 
            data with periodic fluctuations. SARIMA requires careful selection of parameters (seasonal and non-seasonal) but offers flexibility for 
            complex time series.<br>
            <u>Trained on</u> <b>{training_points}</b> data points, requires <b>seasonal and trend parameters</b> for best performance.
            </p>
            ''' , unsafe_allow_html=True
        )
        st.write('About Model Parameters -')
        st.markdown(
            f'''
            <ul style = 'border: 2px solid #888888;padding: 10px;border-radius: 5px;background-color: #2b2b2b;color: #e5e5e5;'>
            <li><strong>Order (Non-seasonal components):</strong>
                <ul>
                <li><strong>(p, d, q):</strong> Represents the non-seasonal ARIMA order.</li>
                <li><strong>p:</strong> Autoregressive (AR) order, which represents the number of lagged observations included in the model. Typically ranging from 0 to 10 or higher, depending on the complexity of the time series data.</li>
                <li><strong>d:</strong> Degree of differencing, which represents the number of times the data needs to be differenced to make it stationary. Any non-negative integer.</li>
                <li><strong>q:</strong> Moving Average (MA) order, which represents the number of lagged forecast errors included in the model. Typically ranging from 0 to 10 or higher, depending on the complexity of the time series data.</li>
                </ul>
            </li>
            <li><strong>Seasonal Order (Seasonal components):</strong>
                <ul>
                <li><strong>(P, D, Q, s):</strong> Represents the seasonal ARIMA order.</li>
                <li><strong>P:</strong> Seasonal autoregressive order.</li>
                <li><strong>D:</strong> Seasonal differencing order.</li>
                <li><strong>Q:</strong> Seasonal moving average order.</li>
                <li><strong>s:</strong> Seasonal period, i.e., the number of time periods in a season.</li>
                </ul>
            </li>
            </ul>
            ''',unsafe_allow_html=True
        )
        s1,s2 = st.columns(2)
        order_input = s1.text_input("Order",help= "eg.1,1,1", placeholder='1,1,1',value = '1,1,1')
        seasonal_order_input = s2.text_input("Seasonal Order",help="eg.1,1,1,13", placeholder='1,1,1,13',value='1,1,1,13')
        
        st.write('')
        st.markdown("<h3 style='text-align: left;'>5. Prophet</h3>", unsafe_allow_html=True)
        st.markdown(
            f'''
            <p style = '
                border: 2px solid #888888;      
                padding: 10px;
                border-radius: 5px;
                background-color: #2b2b2b;      
                color: #e5e5e5;                
            '>
            Developed by Meta, Prophet is tailored for <i>time series with strong seasonal effects</i> and historical data. It can automatically detect seasonality and holidays, making it ideal for sales data with clear patterns over days, months, or years. It’s robust to missing data and adaptable with minimal parameter tuning.<br>
            <u>Trained on</u> <b>{training_points}</b> data points, allows for <b>additional parameters like yearly, weekly, and holiday seasonality</b>.
            </p>
            ''', unsafe_allow_html=True
        )

# This function calculaes forecasted values for Linear Regression. | Also Applied Calenderization -(holiday factor)]
def linear_forecast(df):
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data.loc[:, 'date_numeric'] = (data['date'] - data['date'].min()).dt.days
    X = data['date_numeric']
    y = data['volume']
    coeffs = np.polyfit(X, y, 1)
    #slope, intercept = coeffs
    trend_function = np.poly1d(coeffs)
    future_dates_df = pd.DataFrame({'date':prediction_dates})
    future_dates_df['date'] = pd.to_datetime(future_dates_df['date'], errors='coerce')
    future_dates_df.loc[:, 'date_numeric'] = (future_dates_df['date'] - data['date'].min()).dt.days
    forecast_sales = trend_function(future_dates_df['date_numeric'])
    forecast_data = pd.DataFrame({ 
        'date': future_dates_df['date'],
        'volume': forecast_sales,
        'type': 'linear'  # To identify forecasted data
    })
    forecast_data['date'] = pd.to_datetime(forecast_data['date'], errors='coerce')
    cal_fac = pd.read_parquet('input_data\calenderization.parquet')
    cal = pd.DataFrame({'DATE' : forecast_data['date']})
    cal['MONTH'] = cal['DATE'].dt.to_period('M').dt.to_timestamp()
    cal['DATE'] = cal['DATE'].dt.date
    cal['MONTH'] = cal['MONTH'].dt.date
    cal = cal.merge(cal_fac,on = ['DATE','MONTH'],how = 'left')
    cal = cal.drop(columns='MONTH')
    cal = cal[['DATE','HOLIDAY_FACTOR']]
    cal.columns = ['date','factor']
    cal['date'] = pd.to_datetime(cal['date'], errors='coerce') # for st bug -
    forecast_data = forecast_data.merge(cal,on ='date', how ='left')
    forecast_data['volume'] = forecast_data['volume'] * forecast_data['factor']
    forecast_data.drop(columns='factor',inplace=True)
    forecast_first_date = forecast_data['date'].min()
    forecast_first_week= forecast_data[(forecast_data['date'] == forecast_first_date) & (forecast_data['type'] == 'linear')]
    forecast_first_week = forecast_first_week.copy()
    forecast_first_week['type'] = 'historical'
    forecast_data_overlap = pd.concat([forecast_data, forecast_first_week], ignore_index=True)

    return(forecast_data,forecast_data_overlap)

# For Gaussian Forecast- (to do : learn parameters | add overlap)
def gpr_forecast(df, kernel_type="rbf", prediction_dates=None, alpha=1e-2, length_scale=1.0, periodicity=1.0):
    """
    Gaussian Process Regression for sales forecasting.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns ['date', 'volume'].
    - kernel_type (str): Kernel type ('rbf', 'periodic', 'rational_quadratic', 'both').
    - prediction_dates (list of datetime.date): Dates to predict sales for.
    - alpha (float): Noise level (regularization) parameter.
    - length_scale (float): Initial length scale parameter for RBF and Rational Quadratic kernels.
    - periodicity (float): Periodicity parameter for the periodic kernel.

    Returns:
    - pd.DataFrame: DataFrame with columns ['date', 'volume'], including predictions.
    """
    df = df.copy()
    # Convert dates to ordinal values for model input
    df['date_ordinal'] = pd.to_datetime(df['date']).map(lambda date: date.toordinal())
    X_train = df['date_ordinal'].values.reshape(-1, 1)
    y_train = df['volume'].values

    # Kernel selection
    if kernel_type == "rbf":
        kernel = RBF(length_scale=length_scale)
    elif kernel_type == "periodic":
        kernel = ExpSineSquared(length_scale=length_scale, periodicity=periodicity)
    elif kernel_type == "rational_quadratic":
        kernel = RationalQuadratic(length_scale=length_scale, alpha=1.0)
    elif kernel_type == "both":
        kernel = RBF(length_scale=length_scale) + ExpSineSquared(length_scale=length_scale, periodicity=periodicity)
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf', 'periodic', 'rational_quadratic', or 'both'.")

    # Define Gaussian Process Regressor with the selected kernel
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)

    # Fit the model
    gpr.fit(X_train, y_train)

    # Prepare prediction dates
    prediction_dates_ordinal = np.array([d.toordinal() for d in prediction_dates]).reshape(-1, 1)

    # Predict volumes for the given future dates
    y_pred, y_std = gpr.predict(prediction_dates_ordinal, return_std=True)

    # Prepare output DataFrame
    predictions_df = pd.DataFrame({
        'date': prediction_dates,
        'volume': y_pred,
        'type': 'gaussian'
    })

    return predictions_df

# for Sarima Forecast- (to do : learn parameters | add overlap)
def sarima_forecast(df,prediction_dates,order = [1,1,1],seasonal_order=[1, 1, 1,13]):
    df = df.copy()
    # Ensure the date column is in datetime format and set it as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Define SARIMA parameters (adjust these through experimentation or use auto_arima)
    p, d, q = [*order] # Basic ARIMA terms
    P, D, Q, s = [*seasonal_order] # Seasonal terms with weekly seasonality

    # Fit SARIMA model
    sarima_model = SARIMAX(df['volume'], order=(p, d, q), seasonal_order=(P, D, Q, s))
    sarima_result = sarima_model.fit(disp=False)

    # Forecast next 13 weeks
    forecast = sarima_result.get_forecast(steps=forecast_points)
    forecast_values = forecast.predicted_mean

    # Optionally, convert forecast to a DataFrame for easier analysis with prediction_dates
    forecast_df = pd.DataFrame({'date': prediction_dates, 'volume': forecast_values,'type':'sarima'}).reset_index(drop=True)

    return (forecast_df)

# for Exponential Forecast  - (to do : learn parameters | add overlap)
def exponential_forecast(df, prediction_dates, trend='add', seasonal='add', seasonal_periods=4, confidence_interval=0.95):
    """
    Forecast future values using Exponential Smoothing.

    Parameters:
    - df (pd.DataFrame): Input dataframe with columns 'date' and 'volume'.
    - prediction_dates (list): List of future dates for forecasting.
    - trend (str): Type of trend component. Options are 'add', 'mul', or None.
    - seasonal (str): Type of seasonal component. Options are 'add', 'mul', or None.
    - seasonal_periods (int): Length of the seasonality cycle.
    - confidence_interval (float): Confidence level for forecast intervals, between 0 and 1.

    Returns:
    - pd.DataFrame: Forecasted values with confidence intervals X 3.
    """
    df = df.copy()
    model = ExponentialSmoothing(
        df['volume'],trend=trend, seasonal_periods=seasonal_periods,seasonal=seasonal
    )
    fitted_model = model.fit()
    future_predictions_exponential = fitted_model.forecast(len(prediction_dates))
    residuals = df['volume'] - fitted_model.fittedvalues
    std_error = np.std(residuals, ddof=1)  # Sample standard deviation
    n = len(df['volume'])
    if trend is not None:
        p = 2 if seasonal is not None else 1
    else:
        p = 1 if seasonal is not None else 0
    df = n - p
    t_stat = t.ppf(1 - confidence_interval / 2, df)
    lower_bound = future_predictions_exponential - t_stat * std_error * np.sqrt(1 + 1 / n)
    upper_bound = future_predictions_exponential + t_stat * std_error * np.sqrt(1 + 1 / n)
    forecast_df = pd.DataFrame({'date': prediction_dates,'volume': future_predictions_exponential,'type' :'exponential'}).reset_index(drop=True)
    forecast_df_low = pd.DataFrame({'date': prediction_dates,'volume': lower_bound,'type' :'exponential_lb'}).reset_index(drop=True)
    forecast_df_up = pd.DataFrame({'date': prediction_dates,'volume': upper_bound,'type' :'exponential_ub'}).reset_index(drop=True)

    return(
        forecast_df,forecast_df_low,forecast_df_up
    )

# For Prohet Forecast
def prophet_forecast(df):
    df = df.copy()
    df.rename(columns={'date': 'ds', 'volume': 'y'}, inplace=True)
    model = Prophet()
    model.fit(df)
    future = pd.DataFrame(prediction_dates, columns=['ds'])
    
    forecast = model.predict(future)
    
    forecast_main = forecast[['ds', 'yhat']].copy()
    forecast_main = forecast_main.rename(columns = {'ds':'date','yhat':'volume'})
    forecast_main['type'] = 'prohet'

    forecast_lb = forecast[['ds', 'yhat_lower']].copy()
    forecast_lb = forecast_lb.rename(columns = {'ds':'date','yhat_lower':'volume'})
    forecast_lb['type'] = 'prohet_lb'

    forecast_ub = forecast[['ds', 'yhat_upper']].copy()
    forecast_ub = forecast_ub.rename(columns = {'ds':'date','yhat_upper':'volume'})
    forecast_ub['type'] = 'prohet_ub'
    
    return(
        forecast_main,forecast_lb,forecast_ub
    )

# validate model params , store in correct manner, manage flags -
def model_param_util():
    
    if st.button('PROCESS DATA',use_container_width=True):
        # EXPONENTIAL #
        ss['trend_option'] = trend_option
        ss['seasonality_option'] = seasonality_option
        ss['seasonal_period'] = seasonal_period
        ss['confidence_interval'] = confidence_interval

        # GAUSSIAN #
        ss['kernel_type'] = kernel_type
        ss['gausian_alpha'] = gausian_alpha
        ss['length_scale'] = length_scale
        ss['periodicity'] = periodicity

        # SARIMA # 
        ss['order_input'] = [int(x) for x in order_input.split(",")]
        ss['seasonal_order_input'] = [int(x) for x in seasonal_order_input.split(",")]
        # st.write('PARMS - ')
        # for var in['trend_option','seasonality_option','seasonal_period','confidence_interval','kernel_type','gausian_alpha','length_scale','periodicity','order_input','seasonal_order_input']:
        #     st.write(var,ss[var])

        # Models are loaded in memory - 
        ss.model_in = True
    st.markdown('---')

# call the functions to make dataframes - uses the params in session state - happens after model_param_util()
def create_dataframes():
    #linear -
    linear_df,linear_ovlp_df = linear_forecast(weekly)

    # Exponential - Working | need to understand params
    exponential_df,exponential_lb_df,exponential_ub_df = exponential_forecast(
        weekly,prediction_dates,
        trend = ss['trend_option'],
        seasonal= ss['seasonality_option'],
        seasonal_periods = ss['seasonal_period'],
        confidence_interval = ss['confidence_interval']
    )
    # Gaussian - Working | need to understand params
    gaussian_df = gpr_forecast(
        weekly, 
        kernel_type=ss['kernel_type'], 
        prediction_dates=prediction_dates, 
        alpha=ss['gausian_alpha'], 
        length_scale=ss['length_scale'], 
        periodicity=ss['periodicity']
    )

    # Sarima - Working | need to understand params
    sarima_df = sarima_forecast(
        weekly,prediction_dates,
        order = ss['order_input'],
        seasonal_order = ss['seasonal_order_input']
    )

    # Prophet 
    prophet_df,prophet_lb_df,prophet_ub_df = prophet_forecast(weekly)

    # aggregate all the resultant dataframes
    df = pd.concat(
        [
            weekly,
            linear_df,
            exponential_df,exponential_lb_df,exponential_ub_df,
            gaussian_df,
            sarima_df,
            prophet_df,prophet_lb_df,prophet_ub_df
        ], 
        ignore_index=True
    )

    return (df)

    #

# graph util 1
def graph_util1(df):
    df = df.copy()
    selected_models = st.multiselect(
        'Select Models',
        options = ['LINEAR','EXPONENTIAL','GAUSSIAN','SARIMA','PROPHET'],
        default=  ['LINEAR','EXPONENTIAL']
    )
    type_filter = ['historical']
    name_type_mapping = {
        'LINEAR' : ['linear'],
        'EXPONENTIAL' : ['exponential','exponential_lb','exponential_ub'],
        'GAUSSIAN' : ['gaussian'],
        'SARIMA' : ['sarima'],
        'PROPHET': ['prohet','prohet_lb','prohet_ub']
    }
    for i in selected_models:
        type_filter.extend(name_type_mapping[i])

    df  = df[df['type'].isin(type_filter)]
    
    fig = px.line(
        df, 
        x='date', y='volume', color='type', 
        #line_group='Territory_Number', 
        title='Volume Forecast',
        #color_discrete_map=colormap, 
        line_dash='type',  # Map line style to 'week_group'
        #line_dash_map=linestylemap,
        labels={'volume' : 'Volume','date' : ''},
        #width=1000,
        height=800,
    )
    fig.update_layout(
        title_font_size=40,
        title = {'x': 0.5,'xanchor': 'center', 'yanchor': 'top' },
        legend = dict(
            font = dict(size=18),
            title = 'Source',
            title_font = dict(size = 22),
        )
    )
    fig.update_yaxes(title_font = dict(size = 36),tickfont = dict(size = 18))
    fig.add_shape(type="rect",xref="paper", yref="paper",x0=0, y0=0, x1=1, y1=1,line=dict(color="black", width=2))

    g1, g2, g3 = st.columns((1, 5, 0.5))
    st.plotly_chart(fig,use_container_width=True)



#   #   #   #   #   #   #   #
# Main - 
if ss.file_up: #If input file  has been uploaded then-
    
    st.markdown("<h1 style='text-align: center;'>Forecasting</h1>", unsafe_allow_html=True)
    # Temper training and forecast range.
    get_ranges()
    # Input Model Parameters - WIP
    get_model_mods()
    # Validate params | Store | Trigger | Process -
    model_param_util()

    # if parameters have been stored once-
    if ss.model_in == True:
        mega_df = create_dataframes()
        graph_util1(mega_df)