import streamlit as st
import pandas as pd
from sklearn.preprocessing import Imputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier

st.title('Data Preprocessing App')

def get_data():
  st.subheader('Step 1: Get Data')
  data_url = st.text_input('Enter a URL to a CSV file or upload a file')
  if data_url:
    try:
      df = pd.read_csv(data_url)
      st.dataframe(df.head())
      return df
    except Exception as e:
      st.warning('There was an error loading the data: {}'.format(e))
  else:
    uploaded_file = st.file_uploader('Upload a CSV file', type='csv')
    if uploaded_file is not None:
      df = pd.read_csv(uploaded_file)
      st.dataframe(df.head())
      return df

df = get_data()

def impute_data(df):
  st.subheader('Step 2: Impute Missing Values')
  imputation_method = st.selectbox('Select an imputation method', ['Mean', 'Median', 'Most Frequent'])
  if imputation_method == 'Mean':
    imputer = Imputer(strategy='mean')
  elif imputation_method == 'Median':
    imputer = Imputer(strategy='median')
  elif imputation_method == 'Most Frequent':
    imputer = Imputer(strategy='most_frequent')
  else:
    st.warning('Invalid imputation method')
    return df
  df_imputed = imputer.fit_transform(df)
  st.write('Imputation complete')
  return df_imputed

df_imputed = impute_data(df)

# Step 4: Select Features
def select_features(df):
  st.subheader('Step 4: Select Features')
  feature_selection_method = st.selectbox('Select a feature selection method', ['K Best', 'Recursive Feature Elimination'])
  if feature_selection_method == 'K Best':
    num_features = st.slider('Select the number of features to keep', min_value=1, max_value=df.shape[1])
    selector = SelectKBest(chi2, k=num_features)
  elif feature_selection_method == 'Recursive Feature Elimination':
    num_features = st.slider('Select the number of features to keep', min_value=1, max_value=df.shape[1])
    from sklearn.feature_selection import RFE
    model = RandomForestClassifier()
    rfe = RFE(model, num_features)
    selector = rfe
  else:
    st.warning('Invalid feature selection method')
    return df
  df_selected = selector.fit_transform(df, target)
  st.write('Feature selection complete')
  return df_selected

df_selected = select_features(df_imputed)

# Step 5: Oversample Data
def oversample_data(df):
  st.subheader('Step 5: Oversample Data')
  oversample = st.radio('Oversample minority class?', ['Yes', 'No'])
  if oversample == 'Yes':
    smote = SMOTE()
    df_oversampled, target_oversampled = smote.fit_sample(df, target)
    st.write('Oversampling complete')
    return df_oversampled, target_oversampled
  else:
    return df, target

df_oversampled, target_oversampled = oversample_data(df_selected)

# Step 6: Train Classifier
def train_classifier(df, target):
  st.subheader('Step 6: Train Classifier')
  classifier_type = st.selectbox('Select a classifier', ['Random Forest', 'Logistic Regression'])
  if classifier_type == 'Random Forest':
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
  elif classifier_type == 'Logistic Regression':
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
  else:
    st.warning('Invalid classifier type')
    return
  classifier.fit(df, target)
  st.write('Classifier training complete')
  return classifier

classifier = train_classifier(df_oversampled, target_oversampled)

# Step 7: Test Classifier
def test_classifier(classifier, df, target):
  st.subheader('Step 7: Test Classifier')
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=0)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write('Test set accuracy: {:.2f}%'.format(accuracy*100))

test_classifier(classifier, df_oversampled, target_oversampled)