#!/usr/bin/env python
# coding: utf-8

# In[2]:


# len(['annual_inc', 'dti', 'earliest_cr_line', 'emp_length', 'installment',
#        'int_rate', 'pub_rec_bankruptcies', 'revol_bal', 'revol_util',
#        'total_acc', 'issue_year', 'delinq_2yrs', 'fico_avg',
#        'application_type_Joint App', 'home_ownership_OWN',
#        'home_ownership_RENT', 'purpose_home_improvement',
#        'purpose_small_business', 'term_ 60 months', 'region_NorthEast',
#        'region_West', 'purpose_g_life_event'])


# In[ ]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('xgb_model_final.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
    return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(annual_inc, dti, earliest_cr_line, emp_length, installment, int_rate, pub_rec_bankruptcies, revol_bal, revol_util,total_acc, issue_year, delinq_2yrs, fico_avg, application_type_Joint_App, home_ownership_OWN, home_ownership_RENT,purpose_home_improvement, purpose_small_business, term_60_months, region_NorthEast, region_West, purpose_g_life_event):

    prediction = classifier.predict(pd.DataFrame([[annual_inc, dti, earliest_cr_line, emp_length, installment, int_rate, pub_rec_bankruptcies, revol_bal, revol_util,total_acc, issue_year, delinq_2yrs, fico_avg, application_type_Joint_App, home_ownership_OWN, home_ownership_RENT,purpose_home_improvement, purpose_small_business, term_60_months, region_NorthEast, region_West, purpose_g_life_event]]))
    print(prediction)
    return prediction
    

# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Loan Default Prediction")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Loan Defaulter Prediction ML App </h1>
    </div>
    """

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    annual_inc = st.number_input("annual_inc",value = 60000)
    dti = st.number_input("dti",value = 28.3)
    earliest_cr_line = st.number_input("earliest_cr_line",value = 1991)
    emp_length = st.number_input("emp_length",value = 11)
    installment = st.number_input("installment",value=674.37)
    int_rate = st.number_input("int_rate",value = 13.05)
    pub_rec_bankruptcies = st.number_input("pub_rec_bankruptcies", value = 0)
    revol_bal = st.number_input("revol_bal",value = 22436)
    revol_util = st.number_input("revol_util",value = 63.6)
    total_acc = st.number_input("total_acc", value = 37)
    issue_year = st.number_input("issue_year", value = 2013)
    delinq_2yrs = st.number_input("delinq_2yrs",value = 0)
    fico_avg = st.number_input("fico_avg", value = 632)
    application_type_Joint_App = st.number_input("application_type_Joint_App", value = 0)
    home_ownership_OWN = st.number_input("home_ownership_OWN", value =0)
    home_ownership_RENT = st.number_input("home_ownership_RENT", value =0)
    purpose_home_improvement = st.number_input("purpose_home_improvement",value = 0)
    purpose_small_business = st.number_input("purpose_small_business",value = 0)
    term_60_months = st.number_input("term_ 60_months", value=0)
    region_NorthEast = st.number_input("region_NorthEast",value=0)
    region_West = st.number_input("region_West",value=0)
    purpose_g_life_event = st.number_input("purpose_g_life_event",value=0)
    result =""

    if st.button("Predict"):
        result = prediction(annual_inc, dti, earliest_cr_line, emp_length, installment, int_rate, pub_rec_bankruptcies, revol_bal, revol_util,total_acc, issue_year, delinq_2yrs, fico_avg, application_type_Joint_App, home_ownership_OWN, home_ownership_RENT,purpose_home_improvement, purpose_small_business, term_60_months, region_NorthEast, region_West, purpose_g_life_event)
    st.success('The output is {}'.format(result))
    
if __name__=='__main__':
    main()

