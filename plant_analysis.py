
import pandas as pd
import numpy as np
import MySQLdb
import calendar as cal
import matplotlib.pyplot as plt

import urllib2
import ast
import pandas as pd
import seaborn


wban_id = 53910

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


# In[74]:

df_weather = temperature(wban_id)


# In[126]:

def pull_elastic_sample():
        plant_id_input = 3443
        default_url = "http://api.eia.gov/series/?api_key=45780D1A92A4F363815C75600ACF5748&series_id="
        series_id = "ELEC.PLANT.CONS_EG_BTU."+ str(plant_id_input) +"-NG-ALL.M"
        response = urllib2.urlopen(default_url+series_id)
        tmp = response.read().replace('null','-1')
        eia_pull = ast.literal_eval(tmp)
        df_plant = pd.DataFrame(eia_pull['series'][0]['data'],columns = ['yearmonth','gen'])
        target = df_plant
        df_value = pd.DataFrame(target['gen'])
        df_year=pd.DataFrame(target['yearmonth'].str[:4])
        df_year.columns.values[0] = 'year'
        df_month=pd.DataFrame(target['yearmonth'].str[-2:])
        df_month.columns.values[0] = 'month'
        df_month.month = df_month.month.astype(int)
        df_year_month=df_month.combine_first(df_year)
        df_target=df_year_month.combine_first(df_value)
        df_target = df_target[['year','month','gen']]
        df_data = df_target.merge(df_weather, how = 'left', on = ['year', 'month'])
        ax = df_data[42:54].plot(kind='scatter', x='Temp', y = 'gen',color='DarkBlue', label='2011',
                                title = 'Plant monthly generation vs Temperature, Plant ID:'+ str(plant_id_input))
        df_data[30:42].plot(kind='scatter', x='Temp', y = 'gen',color='Green', label='2012',ax=ax)
        df_data[18:30].plot(kind='scatter', x='Temp', y = 'gen',color='Red', label='2013',ax=ax)
        df_data[6:18].plot(kind='scatter', x='Temp', y = 'gen',color='Orange', label='2014',ax=ax)
        #plot = ax
        #fig = plot.get_figure()
        #fig.savefig("inelastic.png")
        #return fig


def pull_inelastic_sample():
        plant_id_input = 3452
        default_url = "http://api.eia.gov/series/?api_key=45780D1A92A4F363815C75600ACF5748&series_id="
        series_id = "ELEC.PLANT.CONS_EG_BTU."+ str(plant_id_input) +"-NG-ALL.M"
        response = urllib2.urlopen(default_url+series_id)
        tmp = response.read().replace('null','-1')
        eia_pull = ast.literal_eval(tmp)
        df_plant = pd.DataFrame(eia_pull['series'][0]['data'],columns = ['yearmonth','gen'])
        target = df_plant
        df_value = pd.DataFrame(target['gen'])
        df_year=pd.DataFrame(target['yearmonth'].str[:4])
        df_year.columns.values[0] = 'year'
        df_month=pd.DataFrame(target['yearmonth'].str[-2:])
        df_month.columns.values[0] = 'month'
        df_month.month = df_month.month.astype(int)
        df_year_month=df_month.combine_first(df_year)
        df_target=df_year_month.combine_first(df_value)
        df_target = df_target[['year','month','gen']]
        df_data = df_target.merge(df_weather, how = 'left', on = ['year', 'month'])
        ax = df_data[42:54].plot(kind='scatter', x='Temp', y = 'gen',color='DarkBlue', label='2011',
                                title = 'Plant monthly generation vs Temperature, Plant ID:'+ str(plant_id_input))
        df_data[30:42].plot(kind='scatter', x='Temp', y = 'gen',color='Green', label='2012',ax=ax)
        df_data[18:30].plot(kind='scatter', x='Temp', y = 'gen',color='Red', label='2013',ax=ax)
        df_data[6:18].plot(kind='scatter', x='Temp', y = 'gen',color='Orange', label='2014',ax=ax)
        #plot = ax
        #fig = plot.get_figure()
        #fig.savefig("inelastic.png")
        #return fig
