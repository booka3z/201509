
# coding: utf-8

# In[347]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import MySQLdb
import calendar as cal
import matplotlib.pyplot as plt
from sklearn import neighbors as n
# Function that pulls EIA gas consumption by power plants by state:

def eia_state_burn(state_id_input):

    ##connect to eia db
    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","eiadev" )
    cur= db.cursor()

    ## Pull monthly temperature
    datestring = "'2009-12-31'"
    typestring = "'%electric%'"
    query = ('SELECT eia_data.date, eia_data.value / DATEDIFF(DATE_ADD(eia_data.date,INTERVAL 1 MONTH),eia_data.date)'
                'FROM eiadev.eia_series '
                'JOIN eia_data ON eia_data.eia_series_id = eia_series.eia_series_id '
                'WHERE name LIKE '+str(typestring)+
                    'AND obs_state_id = '+str(state_id_input)+
                    ' AND date >'+str(datestring)+
                'GROUP BY 1 '
                'ORDER BY 1')
    cur.execute(query)
    results = cur.fetchall()

    data = []
    for element in results:
        data.append((element[0:3]))

    df = pd.DataFrame(data, columns = ['Date','StateBurnDaily'])
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['month']=pd.DatetimeIndex(df['Date']).month
    df = df.drop('Date', 1)

    ## Pull state name
    query = ('SELECT name FROM eiadev.obs_state '
             'WHERE id = '+str(state_id_input))
    cur.execute(query)
    results_state = cur.fetchall()

    print "Gas Consumption by Power Burn for %s has been fetched. There are a total of %r records." %(results_state[0][0], len(df))

    db.close()
    return df;


# The function that pulls gas consumption by power plants in mmbtu:
def plant_eg_btu(plant_id_input):

    ## Pull Data from EIA API ##
    default_url = "http://api.eia.gov/series/?api_key=45780D1A92A4F363815C75600ACF5748&series_id="
    series_id = "ELEC.PLANT.CONS_EG_BTU."+ str(plant_id_input) +"-NG-ALL.M"
    eia_pull = pd.read_json(path_or_buf = default_url+series_id, orient = 'columns', typ = 'series')

    ## Organize EIA data into dataframes
    target = pd.DataFrame(eia_pull.series[0]['data'],columns = ['yearmonth',str(plant_id_input)] )
    df_value = pd.DataFrame(target[str(plant_id_input)])
    df_year=pd.DataFrame(target['yearmonth'].str[:4])
    df_year.columns.values[0] = 'year'
    df_month=pd.DataFrame(target['yearmonth'].str[-2:])
    df_month.columns.values[0] = 'month'
    df_month.month = df_month.month.astype(int)
    df_year_month=df_month.combine_first(df_year)
    df_target=df_year_month.combine_first(df_value)
    #print "Data for Power Plant # %.f fetched" %plant_id_input
    df_target = df_target[['year','month',str(plant_id_input)]]

    def f(x):
        return cal.monthrange(int(x[0]), int(x[1]))[1]
    df_target['daysInMonth']=df_target.apply(f, axis=1)
    df_target[str(plant_id_input)]=df_target[str(plant_id_input)]/df_target['daysInMonth']
    df_target= df_target.drop('daysInMonth', 1)

    return df_target;


# The function that pulls the latest datapoint of power plant. This function is used to find the data series that are used in EIA's monthly sample.

def check_plant_maxdate(plant_id_input):

    ## Pull Data from EIA API ##
    default_url = "http://api.eia.gov/series/?api_key=45780D1A92A4F363815C75600ACF5748&series_id="
    series_id = "ELEC.PLANT.CONS_EG_BTU."+ str(plant_id_input) +"-NG-ALL.M"
    eia_pull = pd.read_json(path_or_buf = default_url+series_id, orient = 'columns', typ = 'series')

    target = pd.DataFrame(eia_pull.series[0]['data'],columns = ['yearmonth','value'] )
    print "The latest datapoint for Power Plant # %.f is in %s" %(plant_id_input, target[:1]['yearmonth'][0])
    return target[:1]['yearmonth'][0];


# This function pulls the monthly average price of specific price location:

def price_monthly(price_id_input):

    ##connect to eia db
    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","pricedev" )
    cur= db.cursor()

    ## Pull monthly price
    datestring = "'2009-12-31'"
    typestring = "'%electric%'"
    query = ('SELECT year, month, monthly_price_average FROM pricedev.price_cash_monthly_average '
            'WHERE file_id = '+str(price_id_input))
    cur.execute(query)
    results = cur.fetchall()

    data = []
    for element in results:
        data.append((element[0:3]))

    df = pd.DataFrame(data, columns = ['year','month','price'])

    ## Pull location name
    query = ('SELECT pricing_point FROM pricedev.price_location '
             'WHERE id = '+str(price_id_input))
    cur.execute(query)
    results_state = cur.fetchall()


    print "Monthly Price at  %s has been fetched. There are a total of %r records." %(results_state[0][0], len(df))

    db.close()
    return df;


# This function generates a list of all gas-fired power plants:

def find_all_gas_plant():
    ##connect to power db
    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","powerdev" )
    cur= db.cursor()

    ## Pull all plants that use gas fuel

    query = ('SELECT plant_id,state FROM powerdev.power_main WHERE year >2010 AND fuel_group = "NG" GROUP BY 1 , 2 ')
    cur.execute(query)
    results = cur.fetchall()

    data = []
    for element in results:
        data.append((element[0:2]))

    gasPlantList= pd.DataFrame(data, columns = ['plant_id','state'])

    db.close()
    return gasPlantList;


# This function pulls monthly temperature history of a specific weather station:

def temperature(wban_id_input):

    ##connect to weather db
    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","weatherdev" )
    cur= db.cursor()

    ## Pull monthly temperature
    query = ("SELECT year,month, monthly_Tavg FROM weather_actual_monthly_Tavg WHERE wban_id = "+str(wban_id_input)+" AND year > 2009")
    cur.execute(query)
    results = cur.fetchall()

    data = []
    for element in results:
        data.append((element[0:3]))

    df = pd.DataFrame(data, columns = ['year','month','Temp'])

    ## Pull station name
    query = ('SELECT city, state FROM weatherdev.weather_station '
             'WHERE wban_id = '+str(wban_id_input))
    cur.execute(query)
    results_state = cur.fetchall()

    print "Temperature for station %.f at %s, %s fetched" %(wban_id_input, results_state[0][0],results_state[0][1])

    db.close()
    return df;


# This function compile the list of power plants that EIA uses in monthly samples:

def find_all_monthly_samples():
    latest_date = '201410'
    df3=pd.DataFrame()
    for element in gasPlantList['plant_id']:
        try:
            if check_plant_maxdate(element) == latest_date :
                df3 = df3.append([[element,1]],ignore_index=True)
            else:
                df3 = df3.append([[element,0]],ignore_index=True)
        except AttributeError:
            print "Data retrieval ERROR : Power Plant # %.f" %element
        else:
            None
    df3.columns = ['plant_id','in_sample']

    


# This function generates a list of all gas-fired power plants from propdev DB:
def pull_monthly_sample_list():
    ##connect to prop db
    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","propdev" )
    cur= db.cursor()

    query = ('SELECT * FROM propdev.gas_plant')
    cur.execute(query)
    results = cur.fetchall()

    data = []
    for element in results:
        data.append((element[0:4]))

    gasPlantList= pd.DataFrame(data, columns = ['index','plant_id','monthly_sample','state'])
    gasPlantList = gasPlantList.drop('index',1)

    db.close()
    return gasPlantList;



#This function looks up the state location of a plant
def plant_state(plant_id_input):
    
    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","powerdev" )
    cur= db.cursor()

    query = ('SELECT state FROM powerdev.power_plant WHERE plant_id = '+str(plant_id_input)+' GROUP BY 1')
    cur.execute(query)
    results = cur.fetchall()
    return results[0][0];

# This function returns the state ID of a state
def obs_state(state_code_input):
    """Returns the state_id of a state based on eiadev.obs_state
    
    Args:
      state_code_input (char(2)): Two letter abbreviation of a state. i.g., 'CA' for California, 'TX' for Texas
    
    Returns:
      int: state_id based on obs_state
    """

    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","eiadev" )
    cur= db.cursor()

    statestring = "'"+state_code_input+"'"
    query = ('SELECT id FROM eiadev.obs_state2 WHERE state_code = '+str(statestring)+' GROUP BY 1')
    cur.execute(query)
    results = cur.fetchall()
    db.close()
    return results[0][0];

# Save the compiled information to file:
def save_to_file():
    '''save df_data to file'''
    file_path = '/Users/wei/Dropbox/Observ/power/results/'
    file_name = 'EIA_monthly_sample_'+str(state)+'.csv'
    df_data.to_csv(file_path+file_name, sep='\t', encoding='utf-8')
    

########################
## PLOTTING FUNCTIONS ##
########################

# This function plots each plant's generation behavior against temperature:
def plot_year_over_year_graph():
    get_ipython().magic(u'matplotlib inline')
    for element in df_data.columns[3:(len(df_data.columns)-2)]:
        ax= df_data[12:23].plot(kind='scatter', x='Temp', y=str(element),color='DarkBlue', label='2011', title = 'Plant #'+str(element))
        df_data[24:35].plot(kind='scatter', x='Temp', y=str(element),color='Green', label='2012',ax=ax)
        df_data[36:47].plot(kind='scatter', x='Temp', y=str(element),color='Red', label='2013',ax=ax)
        df_data[48:56].plot(kind='scatter', x='Temp', y=str(element),color='Orange', label='2014',ax=ax)
        
# This function plots each plant's generation behavior against temperature:
def plot_plant_against_state(plant_id_input):
    get_ipython().magic(u'matplotlib inline')

    fig, ax = plt.subplots()
    df2=df_data['StateBurnDaily'].astype(float)
    ax.plot(df2, 'b', label='State')

    ax2 = ax.twinx()
    ax2.plot(df_data[str(plant_id_input)], 'g' , label='Plant #'+str(plant_id_input))

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc='upper right')

    ax.set_title('Fuel Consumption Comparison')



# In[232]:

# This function pulls monthly temperature history of a specific weather station:

def temperature_analogue(wban_id_input, analogue_year_input):

    ##connect to weather db
    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","weatherdev" )
    cur= db.cursor()

    ## Pull monthly temperature
    query = ("SELECT year,month, monthly_Tavg FROM weather_actual_monthly_Tavg WHERE wban_id = "+str(wban_id_input))
    cur.execute(query)
    results = cur.fetchall()

    data = []
    for element in results:
        data.append((element[0:3]))

    df = pd.DataFrame(data, columns = ['year','month','Temp'])
    offset = 2015- analogue_year_input ##### ALERT!! Change 2015 to current year !!!###
    df_offset = df
    df_offset['year']=df['year']+ offset

    ## Pull station name
    query = ('SELECT city, state FROM weatherdev.weather_station '
             'WHERE wban_id = '+str(wban_id_input))
    cur.execute(query)
    results_state = cur.fetchall()

    print "Analogue temperature data for station %.f at %s, %s fetched using year %.f" %(wban_id_input, results_state[0][0],results_state[0][1],analogue_year_input)

    db.close()
    return df_offset;
    


# In[255]:

# This function pulls the forward curve by location:

def pull_forward(price_id_input):

    ##connect to eia db
    server = "eia.cwvrtrnm3ga3.us-west-2.rds.amazonaws.com"
    db = MySQLdb.connect(server,"observ","forecast123","pricedev" )
    cur= db.cursor()

    ## Pull monthly price
    datestring = "'2009-12-31'"
    typestring = "'F'"
    query = ('SELECT contract, value FROM pricedev.price_forward WHERE location = '+str(price_id_input)+
        ' AND type ='+typestring+' AND file_Date = (SELECT MAX(file_date) FROM pricedev.price_forward WHERE location = '+str(price_id_input)+') GROUP BY 1,2')
    cur.execute(query)
    results = cur.fetchall()

    data = []
    for element in results:
        data.append((element[0:2]))

    df = pd.DataFrame(data, columns = ['contract','price'])
    df['year']=pd.DatetimeIndex(df['contract']).year
    df['month']=pd.DatetimeIndex(df['contract']).month
    df=df.drop('contract', 1)
    df = df.dropna()

    ## Pull location name
    query = ('SELECT pricing_point FROM pricedev.price_location '
             'WHERE id = '+str(price_id_input))
    cur.execute(query)
    results_state = cur.fetchall()


    print "Forward Curve at  %s has been fetched. There are a total of %r records." %(results_state[0][0], len(df))

    db.close()
    return df;


# In[189]:

## Set inputs manually by looking up tables in the database:
wban_id = 53910 # weather station id
state= 'TX'
price_id = 42 # from price_county, price_location


# In[190]:

## Generate the plant of list that are in EIA's sample in the state we are focusing on:
df_plant_list = pull_monthly_sample_list();
df_current= df_plant_list[(df_plant_list.state == str(state))&(df_plant_list.monthly_sample == 1)]


# In[191]:

## Pull all EIA, temperature and price data together
df_data = temperature(wban_id).merge(plant_eg_btu(df_current['plant_id'].iloc[0]),how = 'left', on = ['month','year'])
for element in df_current['plant_id'][1:]:
    df_data = df_data.merge(plant_eg_btu(element),how = 'left', on = ['month','year'] )
df_data = df_data.merge(eia_state_burn(obs_state(state) ),how = 'left', on = ['month','year'])
df_data = df_data.merge(price_monthly(price_id),how = 'left', on = ['month','year'])
df_data = df_data.fillna(0)


# In[108]:

## Save the compiled information to file:
#save_to_file()


# In[113]:

## Export plant list for manual markup 
def export_plant_list():
    file_path = '/Users/wei/Dropbox/Observ/power/results/'
    file_name = 'plant_list_'+str(state)+'.csv'
    pd.DataFrame(df_data.columns).to_csv(file_path+file_name, sep='\t', encoding='utf-8')
#export_plant_list()


# In[188]:

## Plot each plant's behavior:
#plot_year_over_year_graph()


# In[118]:

##Enter human judgement in excel and read into python
df_elastic = pd.read_csv('/Users/bocard/Desktop/observ/analysis/plant_list_'+str(state)+'_elastic.csv')
df_elastic=df_elastic.set_index('0');
df_data=df_data.fillna(0);
df_elastic['inelastic']=pd.DataFrame(1-df_elastic['elastic'])
df_elastic['inelastic'].loc['year'] = 0
df_elastic['inelastic'].loc['month'] = 0
df_elastic['inelastic'].loc['StateBurnDaily'] = 0
df_elastic['inelastic'].loc['price'] = 0


# In[283]:

## Calculate each sample's subtotals and calculate regression efficients
df_data['StateBurnDaily']=df_data['StateBurnDaily'].astype(float)
df_sum = df_data.ix[:,0:(len(df_data.columns)-1)].dot(df_elastic[0:(len(df_data.columns)-1)])
df_sum = df_sum.join(df_data[['year','month']])
pd.ols(y=df_data['StateBurnDaily']*1000, x=df_sum)


# In[329]:

df_sum.plot(title = 'Classification')


# In[128]:

# Calculate Regression:
df_data['price']=df_data['price'].astype(float)
pd.ols(y=df_sum['elastic'], x=df_data[['Temp','price']])


# In[129]:

# Calculate Regression:
pd.ols(y=df_sum['inelastic'], x=df_data[['Temp','price']])


# In[273]:

df_plot=pd.DataFrame(df_sum['elastic'])
df_plot = df_plot.join(df_data[['Temp','price']])
ax= df_plot[12:23].plot(kind='scatter', x='Temp', y='elastic',color='DarkBlue', label='2011', title = 'elastic vs temp')
df_plot[24:35].plot(kind='scatter', x='Temp', y='elastic',color='Green', label='2012',ax=ax)
df_plot[36:47].plot(kind='scatter', x='Temp', y='elastic',color='Red', label='2013',ax=ax)
df_plot[48:56].plot(kind='scatter', x='Temp', y='elastic',color='Orange', label='2014',ax=ax)


# In[263]:

T.dtypes


# In[264]:

## Pull Forward and temperature data
df_future=pull_forward(42)
df_future=df_future.merge(temperature_analogue(wban_id, 2007), how = 'left', on = ['year','month'])
df_future['price']=df_future['price'].astype(float)
T = df_future[['Temp','price']]


# In[291]:

df_sum[['inelastic']][:-2]


# In[349]:

# Fit regression model for inelastic group
y=df_sum[['inelastic']][:-2]
X=df_data[['Temp','price']][:-2]
T=df_future[['Temp','price']]
neigh = n.KNeighborsRegressor(n_neighbors=3)
y_inelastic=neigh.fit(X, y).predict(T)
df_forecast_i=pd.DataFrame(y_inelastic).join(df_future[['year','month']])
df_forecast_i.columns = ['inelastic', 'year', 'month']


# In[348]:

# Fit regression model for elastic group
y=df_sum[['elastic']][:-2]
X=df_data[['Temp','price']][:-2]
T=df_future[['Temp','price']]
n_neighbors = 5
neigh = n.KNeighborsRegressor(n_neighbors=3)
y_elastic=neigh.fit(X, y).predict(T)
df_forecast_e=pd.DataFrame(y_elastic).join(df_future[['year','month']])
df_forecast_e.columns = ['elastic', 'year', 'month']


# In[327]:

df_forecast=df_forecast_e.merge(df_forecast_i, how='left', on = ['year','month'])


# In[328]:

## build load growth -- Go through new plant built information to decide what the load growth is
#growth

