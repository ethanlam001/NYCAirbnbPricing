from flask import Flask, render_template, request
import pickle
import numpy as np 

app=Flask(__name__)

loaded_model = pickle.load(open("nycairbnbmodel.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,49)
    result = loaded_model.predict(to_predict)
    result = np.e** result[0] -1
    return '$'+str(round(result,2))

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        #print(to_predict_dict.values())
        to_predict_list= list(to_predict_dict.values())
        print(to_predict_dict)
        # Neighbourhood 
        neighbourhood = ['Bedford-Stuyvesant', 'Bushwick', 'Chelsea', 'Clinton Hill','Crown Heights', 
        'East Flatbush', 'East Harlem', 'East Village', 'Financial District', 'Flatbush', 'Flushing', 
        'Fort Greene', 'Greenpoint', 'Harlem', "Hell's Kitchen", 'Kips Bay', 'Long Island City', 
        'Lower East Side', 'Midtown', 'Murray Hill', 'Park Slope', 'Prospect-Lefferts Gardens', 
        'Ridgewood', 'Sunset Park', 'Upper East Side', 'Upper West Side', 'Washington Heights',
        'West Village', 'Williamsburg', 'other']
        neighbourhood_list = [1 if i == to_predict_dict['neighbourhood'] else 0 for i in neighbourhood]
        # Neighbourhood Groups
        neighbourhood_group = ['Bronx','Brooklyn', 'Manhatten', 'Queens', 'Staten Island']
        neighbourhood_group_list = [1 if i == to_predict_dict['neighbourhood_group'] else 0 for i in neighbourhood_group]
        # Room Types
        room_type = ['Entire Home/Apartment','Private Room', 'Shared Room']
        room_type_list = [1 if i == to_predict_dict['room_type'] else 0 for i in room_type]
        # For all numerical values 
        to_predict_list1 = neighbourhood_list + neighbourhood_group_list + room_type_list + to_predict_list[3:11]
        to_predict_list1 = list(map(float, to_predict_list1))
        # For Boolean Values
        to_predict_list2 = [True if i == 'Yes' else False for i in to_predict_list[-3:]]
        # Merge the 2 list
        to_predict_list_final = to_predict_list1 + to_predict_list2
        print("Before sending to model", to_predict_list_final)
        print(loaded_model)
        result = ValuePredictor(to_predict_list_final)
        print("result from model", result)
        return render_template("result.html",result=result)

if __name__ == "__main__":
    app.run(debug=True)
