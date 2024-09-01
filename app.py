from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle 
import folium
from map_check  import plot_map

channel_1 = {
    'Area1': (9.955449, 76.266205),
    'Area2': (9.953278, 76.264963),
    'Area3': (9.950183, 76.265216),
    'Area4': (9.947462, 76.266129)
}

channel_2 = {
    'Area5': (9.954675, 76.261415),
    'Area6': (9.952629, 76.262530),
    'Area7': (9.950613, 76.262305),
    'Area8': (9.950055, 76.261305)
}
legend_html = '''
    <div style="position:fixed; bottom:50px; left:50px; z-index:1000; background:white; padding:5px; border:2px solid grey;">
        <p><span style="color:green;">Low Dredging</span></p>
        <p><span style="color:orange;">Medium Dredging</span></p>
        <p><span style="color:red;">High Dredging</span></p>
    </div>
    '''


with open('randomforest.pkl','rb') as p:
    model = pickle.load(p)

# with open('plot_map.pkl','rb') as p:
#     plot_map = pickle.load(p)

# df=pd.read_csv("Final_Data.csv")
# # print(df)

# X = df.drop(columns=['Channel', 'Date', 'Siltation_Amount'])
# y = df['Siltation_Amount']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.head())
# model = AdaBoostRegressor()
# model.fit(X_train, y_train)

# from flask import Flask, render_template, request



# import pickle
# import numpy as np

# # model = pickle.load(open('iri.pkl', 'rb'))
# with open('adabost_siltation_amout_prediction.pkl','rb') as p:
#     map = pickle.load(p)
# print(map)
app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')

@app.route('/video')
def video_player():
    return render_template('home.html')

@app.route('/View', methods=['POST'])
def home():
    tidalwave = request.form['tidalwave']
    wave = request.form['wave']
    current_speed = request.form['current_speed']
    dir_flow = request.form['dir_flow']
    dir_req =request.form['dir_req']
    dept = request.form['dept']
    ship = request.form['ship']
    heavy = request.form['heavy']
    channel=request.form['channel']
    # for i in 
    # print(tidalwave)
    ar = np.array([[tidalwave, wave, current_speed, dir_flow, dir_req,dept,ship,heavy,channel]])
    pred = model.predict(ar)
    print(pred)
    m=None
    if(pred >= 0 and pred<=0.7):
       m= plot_map(channel,1,channel_1,channel_2)
    elif(pred>=0.8 and pred<=1.8):
        m= plot_map(channel,2,channel_1,channel_2) 
    else:
       m= plot_map(channel,0,channel_1,channel_2)
    
    return render_template('after.html', data=(pred,plot_map,channel,channel_1,channel_2,tidalwave,current_speed,ship,heavy,wave))
    


if __name__ == "__main__":
    app.run(debug=True)






