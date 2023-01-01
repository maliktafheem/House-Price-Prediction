import streamlit as st
import pandas as pd
import pickle as pk
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

df_processed = "final_data_processed.csv"
Image = 'houses.jpeg'

st.sidebar.image(Image, caption="Project on DWDM", use_column_width=True)
st.title("Arz-i Riz")

menu = ["Price Prediction", "About"]
choices = st.sidebar.selectbox("Menu", menu)

if choices == 'Price Prediction':
    # remove all for rent properties
    df = pd.read_csv(df_processed)
    df = df[df["Property_Purpose"] == "For Sale"]
    df = df.drop(columns=['Property_Purpose'])
    df = df.drop(columns=['Property_Price'])
    City = st.selectbox("Select City", ("Islamabad", "Karachi", "Lahore"))
    Baths = st.number_input("Enter number of Bathrooms",
                            min_value=1, max_value=11, format='%d')
    Bedrooms = st.number_input(
        "Enter number of Rooms", min_value=1, max_value=11, format='%d')
    Area = st.number_input(
        "Enter Area of House (in marla)",step=1., min_value=1.0,max_value=100.0 ,format='%.1f')
    # get a list of all the area names in the selected city
    Get_Area_Names = df[df['City'] == City]['Area_Name'].unique()
    # sort the list
    Get_Area_Names.sort()
    Area_Name = st.selectbox("Select Area Name", Get_Area_Names)
    Area_Name.replace(" ", "_")

    df_dummy = pd.get_dummies(
        df, columns=['Property_Type', 'City', 'Area_Name'])
    df_new = pd.DataFrame(columns=df_dummy.columns)
    df_new.at[0, 'City_'+City] = 1
    df_new.at[0, 'Area_Name_'+Area_Name] = 1
    df_new.at[0, 'Bedrooms'] = Bedrooms
    df_new.at[0, 'Area'] = Area
    df_new.at[0, 'Baths'] = Baths
    df_new.fillna(0, inplace=True)

    submit = st.button('Predict')
    if submit:
        st.success("Prediction Done")
        est = pk.load(open('finalized_model_mlp.sav', 'rb'))
        ans = est.predict(df_new)
        ans = ans.mean()
        ans = int(ans)
        ans = "{:,}".format(ans)
        st.subheader(f"The price is {ans} (PKR) ")
if choices == 'About':
    st.subheader("About Us")
    info = '''
        To summarize, the steps we took in the project were:
        - Collecting data: we scraped data about house prices and other relevant factors from the website Zameen.com
        - Pre-processing the data: we cleaned and pre-processed the data to ensure that it was ready for use in  the machine learning model.
        - Choosing a machine learning model: we selected a machine learning model (Gradient Boosting Regressor) that was suitable for the problem of predicting house prices.
        - Training the model: we used the data to train the model, adjusting its parameters so that it could accurately predict house prices.
        - Evaluating the model: we evaluated the performance of the model by comparing the predicted prices with the actual prices in the data.
        - Fine-tuning the model: If necessary, we made adjustments to the model's hyperparameters or collected additional data to improve its accuracy.
        - Using the model to make predictions: Once the model was performing well, we used it to make predictions about house prices for properties in the area we were interested in.
        - Saving the model: we saved the model so that we could use it later without having to retrain it.
        - Deploying the model: we deployed the model on a web application using Streamlit.

        Project Members:
        - Tafheem Ul Islam
        - Abdul Manan
        - Mohammad Talha Irfan
        - Zaid Bin Tariq
    '''
    st.markdown(info, unsafe_allow_html=True)
