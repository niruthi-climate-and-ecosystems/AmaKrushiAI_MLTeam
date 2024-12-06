#Import Libraries
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from translatepy import Translator
from pydub import AudioSegment
from pydub.utils import which
import requests
import pandas as pd

# Load the dataset
df = pd.read_excel('Example_advisories.xlsx')

# Separate LabelEncoders for each categorical column
crop_stage_encoder = LabelEncoder()
df['Crop Stage'] = crop_stage_encoder.fit_transform(df['Crop Stage'])

cat_event_encoder = LabelEncoder()
df['Any Cat Event'] = cat_event_encoder.fit_transform(df['Any Cat Event'])

# Prepare the features and target variable
X = df.drop('Agro Advisory', axis=1)
y = df['Agro Advisory']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Streamlit UI
st.title("Ama Krushi Predictions")
st.write(" Agro Advisories by Niruthi Climate and Eco Systems.")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    temp_last = st.slider(
        'Temperature Last Week', 
        int(X['Temperature Last Week'].min()), 
        int(X['Temperature Last Week'].max()),
        key='temp_last'
    )

with col2:
    temp_this = st.slider(
        'Temperature This Week', 
        int(X['Temperature This Week'].min()), 
        int(X['Temperature This Week'].max()),
        key='temp_this'
    )

with col3:
    temp_next = st.slider(
        'Temperature Next Week', 
        int(X['Temperature Next Week'].min()), 
        int(X['Temperature Next Week'].max()),
        key='temp_next'
    )

with col1:
    perc_last = st.slider('Precipitation Last Week', int(X['Precipitation Last Week'].min()), int(X['Precipitation Last Week'].max()),key='perc_last')
with col2:
    perc_this = st.slider('Precipitation This Week', int(X['Precipitation This Week'].min()), int(X['Precipitation This Week'].max()),key='perc_this')
with col3:
    perc_next = st.slider('Precipitation Next Week', int(X['Precipitation Next Week'].min()), int(X['Precipitation Next Week'].max()), key='perc_next')

with col1:
    hum_last = st.slider('Humidity Last Week', int(X['Humidity Last Week'].min()), int(X['Humidity Last Week'].max()), key='hum_last')
with col2:
    hum_this = st.slider('Humidity This Week', int(X['Humidity This Week'].min()), int(X['Humidity This Week'].max()), key='hum_this')
with col3:
    hum_next = st.slider('Humidity Next Week', int(X['Humidity Next Week'].min()), int(X['Humidity Next Week'].max()), key='hum_next')

# Dropdowns for categorical input
option1 = st.selectbox(
    "What is the crop stage?",
    crop_stage_encoder.classes_
)
encoded_option1 = crop_stage_encoder.transform([option1])[0]

option2 = st.selectbox(
    "Any cat event?",
    cat_event_encoder.classes_
)
encoded_option2 = cat_event_encoder.transform([option2])[0]

# Prepare input data and make predictions
input_data = [[temp_last, temp_this, temp_next, perc_last, perc_this, perc_next, hum_last, hum_this, hum_next, encoded_option1, encoded_option2]]
prediction = clf.predict(input_data)

if st.button("Submit",type="primary"):
    st.write(f"The predicted Advisory is: {prediction[0]}")
    def translate_text(text, target_language):
        translator = Translator()
        translation = translator.translate(text, target_language)
        return translation.result

    target_language = "or"  
    translated_text = translate_text(prediction[0], target_language)
    st.write("Odia Translated Text:", translated_text)

    url = "https://audio.dubverse.ai/api/tts"
    payload = {
        "text": translated_text,
        "speaker_no": 1190,
        "config": {"use_streaming_response": False},
    }
    headers = {
        "X-API-KEY": "pgQBpPbAi9YhME75ecoXnRkl9Iof4GCo",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        audio_file = "output.wav"
        with open(audio_file, "wb") as f:
            f.write(response.content)

        sound = AudioSegment.from_wav(audio_file)
        mp3_file = "output.wav"
        st.write("Odia Translated:")
        st.audio(mp3_file, format="audio/wav")
        # st.success("Audio generated and displayed successfully!")
    else:
        st.error(f"Audio generation failed: {response.status_code}, {response.text}")














# Display model accuracy
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# st.write(f"Model Accuracy: {accuracy:.2f}")
