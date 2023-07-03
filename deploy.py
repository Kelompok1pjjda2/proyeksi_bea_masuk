import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.model_selection import validation_curve, LeaveOneOut, train_test_split, cross_val_score
from sklearn.model_selection import cross_validate, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder 
from matplotlib import pyplot
import pickle
import math

#Baca Data
data = pd.read_excel("./SUMMARY DATA SOURCE.xlsx")

loaded_model = pickle.load(open('./model.sav', 'rb'))

LOGO_IMAGE = "./logo.jpeg"
#Disable Warning
st.set_option('deprecation.showPyplotGlobalUse', False)
#Set Size
sns.set(rc={'figure.figsize':(8,8)})
#Coloring
colors_1 = ['#66b3ff','#99ff99']
colors_2 = ['#66b3ff','#99ff99']
colors_3 = ['#79ff4d','#4d94ff']
colors_4 = ['#ff0000','#ff1aff']
st.markdown(
    f"""
    <div style="text-align: center;">
    <img class="logo-img" src="data:png;base64,{base64.b64encode(open(LOGO_IMAGE, 'rb').read()).decode()}">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center; color: #243A74; font-family:sans-serif'>PROYEKSI PENERIMAAN BEA MASUK</h1>", unsafe_allow_html=True)
menu = st.sidebar.selectbox("Select Menu", ("Dashboard", "Prediksi"))

if menu == "Prediksi":

    st.write(data.head(2))
    tahun= st.selectbox("Pilih tahun",data['TAHUN'].unique())
    for item1 in data['TAHUN'].unique():
        if item1 == tahun:
            st.write(' Tahun yang dipilih adalah ', str(tahun))

 
    bulan= st.selectbox("Pilih bulan",data['BULAN'].unique())
    for item2 in data['BULAN'].unique():
        if item2 == bulan:
            st.write(' Bulan yang dipilih adalah ', str (bulan))


    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>INFLASI</p>", unsafe_allow_html=True)
        input_inf = st.number_input('Nilai inflasi', key=1, value= data[(data['BULAN'] == bulan) & (data['TAHUN'] == tahun)]['INFLASI'].values[0])
        for item3 in data['INFLASI'].unique():
            if item1 == tahun and (item2 == data['BULAN'] == bulan):
                st.write(input_inf)
       
        
        st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>ARAMCO (AVG)  </p>", unsafe_allow_html=True)
        input_aramco= st.number_input('ARAMCO (AVG)',key=2,value= data[(data['TAHUN'] == tahun) & (data['BULAN'] == bulan)]['ARAMCO (AVG)'].values[0])
        for item4 in data['ARAMCO (AVG)'].unique():
            if item1 == tahun and (item2 == data['BULAN'] == bulan):
                st.write(input_aramco)
    with col2:
        st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>KURS TENGAH BI (AVG)</p>", unsafe_allow_html=True)
        input_kurs = st.number_input('KURS TENGAH BI (AVG)',key=3, value= data[(data['TAHUN'] == tahun) & (data['BULAN'] == bulan)]['KURS TENGAH BI (AVG)'].values[0])
        for item3 in data['KURS TENGAH BI (AVG)'].unique():
            if item1 == tahun and (item2 == data['BULAN'] == bulan):
                st.write(input_kurs)
        
        
        st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>GDP</p>", unsafe_allow_html=True)
        input_gdp = st.number_input('GDP',key=4,value= data[(data['TAHUN'] == tahun) & (data['BULAN'] == bulan)]['GDP'].values[0])
        for item4 in data['GDP'].unique():
            if item1 == tahun and (item2 == data['BULAN'] == bulan):
                st.write(input_gdp)
        
    st.write("Penerimaan Bea Masuk Tahun ",str(tahun), 'Bulan ', str(bulan))
    st.write("## ", "Rp", data[(data['TAHUN'] == tahun) & (data['BULAN'] == bulan)]['BEA MASUK'].values[0])

    if bulan == "Januari":
            bulandepan = "Februari"
    if bulan == "Februari":
            bulandepan = "Maret"
    if bulan == "Maret":
            bulandepan = "April"
    if bulan == "April":
            bulandepan = "Mei"
    if bulan == "Mei":
            bulandepan = "Juni"
    if bulan == "Juni":
            bulandepan = "Juli"
    if bulan == "Juli":
            bulandepan = "Agustus"
    if bulan == "Agustus":
            bulandepan = "September"
    if bulan == "September":
            bulandepan = "Oktober"
    if bulan == "Oktober":
            bulandepan = "November"
    if bulan == "November":
            bulandepan = "Desember"
    if bulan == "Desember":
            bulandepan = "Januari"    

    # for item10 in data['BULAN'].unique():
    #     if item10 == bulan:
    #         # st.write (' Bulan depan ', str (bulandepan))

     
    
    if st.button("Prediksi"):
        st.write("## Prediksi Sukses")
        #define X & y
        X = data.drop(['BEA MASUK','no', 'BULAN','TAHUN', 'bulan_tahun'], axis=1)
        y = data['BEA MASUK']
        index=[0]
        # for item in df.index:
        #     if item == tahun:
        #         tahun_enc = df.loc[provinsi].values[0]
        

        df_1_pred = pd.DataFrame({
            
            'INFLASI' : input_inf,
            'ARAMCO (AVG)' : input_aramco,
            'GDP' : input_gdp,
            'KURS TENGAH BI (AVG)' : input_kurs,
        },index=index)
        #Set semua nilai jadi 0
        df_kosong_1 = X[:1]
        for col in df_kosong_1.columns:
            df_kosong_1[col].values[:] = 0
        list_1 = []
        for i in df_1_pred.columns:
            x = df_1_pred[i][0]
            list_1.append(x)
        #buat dataset baru
        for i in df_kosong_1.columns:
            for j in list_1:
                if i == j:
                    df_kosong_1[i] = df_kosong_1[i].replace(df_kosong_1[i].values,1)  
      

        df_kosong_1['INFLASI'] = df_1_pred['INFLASI']
        df_kosong_1['GDP'] = df_1_pred['GDP']
        df_kosong_1['ARAMCO (AVG)'] = df_1_pred['ARAMCO (AVG)']
        df_kosong_1['KURS TENGAH BI (AVG)'] = df_1_pred['KURS TENGAH BI (AVG)']
        pred_1 = loaded_model.predict(df_kosong_1)
        beamasukpred = data[(data['TAHUN'] == tahun) & (data['BULAN'] == bulandepan)]['BEA MASUK'].values[0]
        pred_selisih = (beamasukpred / pred_1 -1)*100
    
        st.write("Bea Masuk Tahun ",str(tahun), 'Bulan ', str(bulan))
        st.write('{0:.2f}'.format(pred_1[0]))

        st.write('Prediksi Bea Masuk Periode', str(bulandepan))
        st.write("## ", "Rp", '{0:.2f}'.format(beamasukpred))

        st.write('Prediksi Growth Bea Masuk : ')
        st.write('{0:.2f}'.format(pred_selisih[0]), "%")



