#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[168]:


weather = pd.read_csv('weatherAUS.csv')


# In[169]:


weather.info()


# In[170]:


weather


# In[171]:


weather = pd.DataFrame(weather)
weather.dropna(subset=['RainToday','RainTomorrow'] ,inplace=True)


# In[172]:


weather.info()


# In[173]:


px.histogram(weather, x = 'Location' , title='location vs rainy days' , color = 'RainToday')


# In[174]:


px.scatter(weather , title='mintemp vs maxtemp' , x = 'MinTemp' , y='MaxTemp' , color = 'RainToday') 


# In[175]:


use_sample = False
sample_fraction = 0.1


# In[176]:


if use_sample:
    weather = weather.sample(frac = sample_fraction).copy()


# In[177]:


weather


# In[178]:


from sklearn.model_selection import train_test_split


# In[179]:


train_val_df , test_df = train_test_split(weather , test_size = 0.2, random_state = 42)
train_df , val_df = train_test_split(train_val_df , test_size = 0.25 , random_state = 42)


# In[180]:


print("train_df.shape :", train_df.shape)
print("val_df shape :" , val_df.shape)
print("test_df shape :", test_df.shape)


# In[181]:


plt.title('No. of rows per year')
sns.countplot(x = pd.to_datetime(weather.Date).dt.year);


# In[182]:


year = pd.to_datetime(weather.Date).dt.year
train_df = weather[year < 2015]
val_df = weather[year == 2015]
test_df = weather[year > 2015]


# In[183]:


print("train_df shape :" , train_df.shape)
print("val_df shape :" , val_df.shape)
print("test_df shape :", test_df.shape)


# In[184]:


plt.title("No of rows per year")
sns.countplot(x=pd.to_datetime(weather.Date).dt.year);


# In[185]:


input_cols = list(train_df.columns)[1:-1]
target_cols = 'RainTomorrow'


# In[186]:


print(input_cols , " ", target_cols)


# In[187]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_cols].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_cols].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_cols].copy()


# In[188]:


numeric_cols = train_inputs.select_dtypes(include = np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# In[189]:


train_inputs[numeric_cols].describe()


# In[190]:


train_inputs[categorical_cols].info()


# In[191]:


train_inputs[numeric_cols].nunique()


# In[192]:


train_inputs[categorical_cols].nunique()


# In[193]:


from sklearn.impute import SimpleImputer


# In[194]:


imputer = SimpleImputer(strategy = 'mean')


# In[195]:


weather[numeric_cols].isna().sum()


# In[196]:


train_inputs[numeric_cols].isna().sum()


# In[197]:


imputer.fit(weather[numeric_cols])


# In[198]:


list(imputer.statistics_)


# In[199]:


train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])


# In[200]:


val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])


# In[201]:


test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


# In[202]:


print(train_inputs[numeric_cols].isna().sum().sum())
print(test_inputs[numeric_cols].isna().sum().sum())
print(val_inputs[numeric_cols].isna().sum().sum())


# In[203]:


from sklearn.preprocessing import MinMaxScaler


# In[204]:


scaler = MinMaxScaler()


# In[205]:


scaler.fit(weather[numeric_cols])


# In[206]:


print("Minimum :")
list(scaler.data_min_)


# In[207]:


print("Maximum :")
list(scaler.data_max_)


# In[208]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])


# In[209]:


val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])


# In[210]:


test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# In[211]:


train_inputs[numeric_cols].describe()


# In[212]:


weather[categorical_cols].nunique()


# In[213]:


from sklearn.preprocessing import OneHotEncoder


# In[214]:


#?OneHotEncoder


# In[215]:


encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore',
)


# In[216]:


encoder.fit(weather[categorical_cols])


# In[217]:


encoder.categories_


# In[218]:


encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)


# In[219]:


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols].fillna('Unknown'))


# In[220]:


val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols].fillna('Unknown'))
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols].fillna('Unknown'))


# In[221]:


pd.set_option('Display.max_columns',None)


# In[222]:


train_inputs


# In[223]:


from sklearn.linear_model import LogisticRegression


# In[224]:


get_ipython().run_cell_magic('time', '', "model = LogisticRegression(solver = 'liblinear')\n#?LogisticRegression\n")


# In[225]:


get_ipython().run_cell_magic('time', '', 'model.fit(train_inputs[numeric_cols + encoded_cols] , train_targets)\n')


# In[226]:


print(numeric_cols + encoded_cols)


# In[227]:


print(model.coef_.tolist())


# In[228]:


print(model.intercept_)


# In[229]:


weight_df = pd.DataFrame({
        'feature' : numeric_cols + encoded_cols,
        'Weights' : (model.coef_.tolist())[0]
})


# In[230]:


plt.figure(figsize = (10,50))
sns.barplot(weight_df , x = 'Weights' , y = 'feature')


# In[231]:


sns.barplot(data = weight_df.sort_values('Weights' , ascending = False).head(12) , x = 'Weights' , y = 'feature')


# In[233]:


X_train = train_inputs[numeric_cols+encoded_cols]
X_val = val_inputs[numeric_cols+encoded_cols]
X_test = test_inputs[numeric_cols+encoded_cols]


# In[234]:


pred_model = model.predict(X_train)


# In[235]:


pred_model


# In[236]:


train_targets


# In[237]:


train_probs = model.predict_proba(X_train)
print(train_probs)


# In[238]:


from sklearn.metrics import accuracy_score


# In[239]:


accuracy_score(train_targets , pred_model)


# In[240]:


from sklearn.metrics import confusion_matrix
confusion_matrix(train_targets , pred_model , normalize = 'true')
#?confusion_matrix


# In[241]:


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));
    
    return preds


# In[242]:


train_preds = predict_and_plot(X_train, train_targets, 'Training')


# In[243]:


def random_guess(inputs):
    return np.random.choice(["No", "Yes"], len(inputs))


# In[244]:


def all_no(inputs):
    return np.full(len(inputs), "No")


# In[245]:


accuracy_score(test_targets, random_guess(X_test))


# In[246]:


accuracy_score(test_targets, all_no(X_test))


# In[247]:


new_input = {'Date': '2021-06-19',
             'Location': 'Katherine',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# In[250]:


new_input_df = pd.DataFrame([new_input])


# In[251]:


new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])


# In[252]:


new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])


# In[253]:


new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])


# In[254]:


X_new_input = new_input_df[numeric_cols + encoded_cols]
X_new_input


# In[255]:


prediction = model.predict(X_new_input)[0]


# In[256]:


prediction


# In[257]:


prob = model.predict_proba(X_new_input)[0]


# In[258]:


prob


# In[259]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob


# In[260]:


new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# In[261]:


print(predict_input(new_input))


# In[ ]:




