import streamlit as st
import yaml
import streamlit_authenticator as stauth
import pandas as pd
import pickle
import datetime

# Load the YAML configuration file
with open("allowed_users.yaml") as file:
    config = yaml.safe_load(file)

# Load the trained Voting Classifier model
with open("voting_classifier_ex_xgb_dur_hml_5t.pkl", "rb") as file:
    voting_classifier_ex_xgb_dur_hml_5t = pickle.load(file)

# Load the preprocessor used during training (if applicable)
with open("preprocessor_dur_hml_5t.pkl", "rb") as file:
    preprocessor_dur_hml_5t = pickle.load(file)

# Set cookie expiry to 5 seconds
authenticator = stauth.Authenticate(
    config['credentials'],
    'news_app_cookie_test',  # Replace with your own cookie name
    'abc123',  # Replace with your own signature key
    cookie_expiry_days= 7  # Cookie expires after 5 seconds. a day has 86400 seconds
)

# Add Login Form

login_result = authenticator.login()	

if st.session_state['authentication_status']:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')


    genre_options = sorted([
            "WAR", "ASTROLOGY", "RELIGIOUS / FAITH", "HEALTH", "NATIONAL THREAT/DEFENCE NEWS",
            "INDIA-PAK", "CRIME/LAW & ORDER", "POLITICAL NEWS/GOVERNMENT NEWS", "FINANCIAL NEWS",
            "SCIENCE/SPACE", "CAREER/EDUCATION", "EVENT/CELEBRATIONS", "WEATHER/ENVIRONMENT",
            "ENTERTAINMENT NEWS", "OTHER", "MISHAPS/FAILURE OF MACHINERY", "SPORTS NEWS"
        ])
    
    geography_options = sorted([
            "INTERNATIONAL", "MANIPUR", "JHARKHAND", "GUJARAT", "INDIAN", "HARYANA", "RAJASTHAN",
            "BIHAR", "UTTAR PRADESH", "OTHER", "DELHI", "MAHARASHTRA", "UTTARAKHAND", "KARNATAKA",
            "JAMMU AND KASHMIR", "CHANDIGARH", "WEST BENGAL", "HIMACHAL PRADESH", "MADHYA PRADESH",
            "TELANGANA", "CHHATTISGARH"
        ])
    
    popularity_options = ["H", "M", "L"]
    
    personality_genre_options = sorted([
            "JMM", "Astrologer", "International", "JDU", "Bajrang Dal", "RJD", "Religious",
            "DMK", "BSP", "INC", "AIMIM", "OTHER", "NCP", "Defense", "SP", "BJP", "TMC", "RSS-VHP",
            "AAP", "SS", "Entertainer", "NC", "SBSP", "Cricketer"
        ])
    
    dur_layers_options = ["H", "M", "L"]
    
    substories_layers_options = ["H", "M", "L"]
    
    logistics_options = ["ON LOCATION", "IN STUDIO", "BOTH"]
    
    story_format_options = sorted(["INTERVIEW", "DEBATE OR DISCUSSION", "NEWS REPORT"])
    
    # Streamlit app interface
    st.title("Hindi News Story Rating Prediction based on Machine Learning model")
    
    # Collect user inputs via Streamlit input elements
    genre = st.selectbox("Select Genre", genre_options)
    geography = st.selectbox("Select Geography (For national stories select INDIAN)", geography_options)
    personality_popularity = st.selectbox("Select Personality Popularity", popularity_options)
    personality_genre = st.selectbox("Select Personality-Genre", personality_genre_options)
    dur_layers = st.selectbox("Select Duration layers (HML)", dur_layers_options)
    substories_layers = st.selectbox("Select Substories layers (HML)", substories_layers_options)
    logistics = st.selectbox("Select Logistics", logistics_options)
    story_format = st.selectbox("Select Story Format", story_format_options)
    
    # Create the DataFrame with the collected inputs
    new_data_show_case = pd.DataFrame({
            'Genre': [genre],
            'Geography': [geography],
            'Personality Popularity': [personality_popularity],
            'Personality-Genre': [personality_genre],
            'Dur Hour (HML)': [dur_layers],
            'Substoires layers (HML)': [substories_layers],
            'Logistics': [logistics],
            'Story_Format': [story_format]
        })
    
    # Display the DataFrame in Streamlit app
    st.write("User Input Data:")
    st.dataframe(new_data_show_case)
    
    # Button to trigger prediction
    # Preprocessing: Transform the new data using the preprocessor fitted on the training data
    if st.button("Predict Rating Tier"):
        new_data_transformed_show_case = preprocessor_dur_hml_5t.transform(new_data_show_case)
    
        # Convert the sparse matrix to a dense matrix (if applicable)
        if hasattr(new_data_transformed_show_case, "toarray"):
            new_data_transformed_dense_show_case = new_data_transformed_show_case.toarray()
        else:
            new_data_transformed_dense_show_case = new_data_transformed_show_case
    
        # Make a prediction using the trained voting classifier model
        new_predictions_show_case = voting_classifier_ex_xgb_dur_hml_5t.predict(new_data_transformed_dense_show_case)
    
        # Define the function to categorize Predicted TVTs
        def categorize_tier(tier):
            if tier == 0:
                return 'Minimal viewership'
            elif tier == 1:
                return 'Low viewership'
            elif tier == 2:
                return 'Average viewership'
            elif tier == 3:
                return 'High viewership'
            elif tier == 4:
                return 'Max viewership'
            else:
                return 'Invalid tier'  # Optional: handle unexpected tier values
    
        # Convert the numerical prediction to the categorized tier
        predicted_value_tier = categorize_tier(new_predictions_show_case[0])
    
        # Display the result
        st.write(f"Predicted Rating Category: {predicted_value_tier}")

        note = (
        "The predicted value tier is determined based on a five-point scale, ranging from lowest to highest. "
        "The tiers are categorized as follows:\n"
        "• Minimal Viewership: Less than 213 TVTs\n"
        "• Low Viewership: 213 to 244 TVTs\n"
        "• Average Viewership: 244 to 276 TVTs\n"
        "• High Viewership: 277 to 318 TVTs\n"
        "• Maximum Viewership: 319 TVTs and above.")

        st.write(note)


# Add the explanatory note

elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')


# Add the professional note at the end of the app
st.write("""
---
### Note:
This app leverages machine learning to predict news ratings, offering insights based on historical data. 
Predictions should be combined with domain expertise. The developer is not responsible for outcomes based solely on the app's predictions. 
For technical details on ML models employed and error metrics, contact:  
**Puneet Sah**  
Mobile: 9820615085
Email: puneet.sah@timesgroup.com
""")
