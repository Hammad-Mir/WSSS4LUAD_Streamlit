import cv2
import torch
import predict
import streamlit as st

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = predict.load_model()

st.title('Tissue Segmentation')

img_name = st.sidebar.selectbox(
    'Select Image',
    ('00.png', '01.png', '02.png', '03.png', '04.png', '05.png', '06.png', '07.png', '08.png', '09.png',
     '10.png', '11.png', '12.png', '13.png', '14.png', '15.png', '16.png', '17.png', '18.png', '19.png',
     '20.png', '21.png', '22.png', '23.png', '24.png', '25.png', '26.png', '27.png', '28.png', '29.png',
     '30.png', '31.png', '32.png', '33.png', '34.png', '35.png', '36.png', '37.png', '38.png', '39.png',
     '40.png', '41.png', '42.png', '43.png', '44.png', '45.png', '46.png', '47.png', '48.png', '49.png',
     '50.png', '51.png', '52.png', '53.png', '54.png', '55.png', '56.png', '57.png', '58.png', '59.png',
     '60.png', '61.png', '62.png', '63.png', '64.png', '65.png', '66.png', '67.png', '68.png', '69.png',
     '70.png', '71.png', '72.png', '73.png', '74.png', '75.png', '76.png', '77.png', '78.png', '79.png')
)

image = cv2.imread('images/img/' + str(img_name))

st.write('## Tissue Image')
st.image(image, use_column_width='auto')

click = st.sidebar.button('Predict')

st.write('## Segmented Tissue Image')
if click:

    fig = predict.predict(model, img_name)
    st.pyplot(fig)