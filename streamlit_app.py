import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Intelligent Systems — AI Web App",
    page_icon="🏍️",
    layout="wide",
)

# Try TensorFlow Keras first, fallback to keras package
try:
    from tensorflow.keras.models import load_model as keras_load_model
except (ModuleNotFoundError, ImportError):
    try:
        from keras.models import load_model as keras_load_model
    except (ModuleNotFoundError, ImportError) as e:
        keras_load_model = None
        # We cannot st.error here until app loads
        print(f"ไม่สามารถโหลดโมเดล Keras ได้: {e}")

# sleep model loader
@st.cache_resource
def load_sleep_model():
    model = joblib.load('ensemble_sleep_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

# flower model loader
@st.cache_resource
def load_flower_model():
    if keras_load_model is None:
        st.error('ไม่พบแพ็กเกจ TensorFlow/Keras ในระบบ: ติดตั้ง tensorflow หรือ keras ก่อน')
        return None
    try:
        return keras_load_model('flower_model_final.keras')
    except Exception as e:
        st.error(f"โหลด Flower model ไม่สำเร็จ: {e}")
        return None


def preprocess_flower_image(image: Image.Image, target_size=(128, 128)):
    """Preprocess image for flower model - 128x128 RGB input
    The model has internal Rescaling layer, so we just need proper shape"""
    img = image.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32')
    # Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr



def run_sleep_tab():
    st.subheader('Sleep Quality Prediction')
    
    # Create subtabs for Usage and Details
    usage_tab, details_tab = st.tabs(['การใช้งานโมเดล', 'รายละเอียดของโมเดล'])
    
    with usage_tab:
        model, feature_names = load_sleep_model()

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Age', 10, 80, 30)
            gender = st.selectbox('Gender', ['Male', 'Female'])
            occupation = st.selectbox('Occupation', ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Manager'])
            sleep_dur = st.number_input('Sleep Duration (hours)', 0.0, 12.0, 7.0)
            phys_act = st.slider('Physical Activity Level', 0, 100, 50)

        with col2:
            stress = st.slider('Stress Level', 1, 10, 5)
            bmi = st.selectbox('BMI Category', ['Normal', 'Overweight', 'Obese'])
            bp = st.text_input('Blood Pressure (Systolic/Diastolic)', '120/80')
            heart_rate = st.number_input('Heart Rate', 40, 120, 72)
            steps = st.number_input('Daily Steps', 0, 20000, 5000)
            disorder = st.selectbox('Sleep Disorder', ['None', 'Sleep Apnea', 'Insomnia'])

        if st.button('ทำนายผล', key='sleep_predict'):
            try:
                sys_bp, dia = map(int, bp.split('/'))
                input_df = pd.DataFrame(columns=feature_names)
                input_df.loc[0] = 0
                input_df['Age'] = age
                input_df['Sleep Duration'] = sleep_dur
                input_df['Physical Activity Level'] = phys_act
                input_df['Stress Level'] = stress
                input_df['Heart Rate'] = heart_rate
                input_df['Daily Steps'] = steps
                input_df['Systolic Pressure'] = sys_bp
                input_df['Diastolic Pressure'] = dia

                if f'Gender_{gender}' in input_df.columns:
                    input_df[f'Gender_{gender}'] = True
                if f'Occupation_{occupation}' in input_df.columns:
                    input_df[f'Occupation_{occupation}'] = True
                if f'BMI_{bmi}' in input_df.columns:
                    input_df[f'BMI_{bmi}'] = True

                sd_col = 'SleepDisorder_No Sleep Disorder' if disorder == 'None' else f'SleepDisorder_{disorder}'
                if sd_col in input_df.columns:
                    input_df[sd_col] = True

                raw_prediction = model.predict(input_df)[0]
                penalty = 0
                reasons = []

                if sleep_dur < 7.0:
                    penalty += (7.0 - sleep_dur) * 1.5
                    reasons.append(f'**[วิกฤตระยะเวลานอน]** นอนเพียง {sleep_dur} ชม. ต่ำกว่าเกณฑ์ที่ควรจะเป็น')
                elif sleep_dur > 9.0:
                    penalty += (sleep_dur - 9.0) * 1.0
                    reasons.append(f'**[ระยะเวลานอนมากเกินไป]** นอนถึง {sleep_dur} ชม. อาจทำให้เกิดภาวะเฉื่อยชา (Sleep Inertia)')

                if stress >= 9:
                    penalty += 2.5
                    reasons.append(f'**[วิกฤตความเครียด]** ระดับ {stress}/10 เป็นระดับอันตรายที่ขัดขวางการเข้าสู่สภาวะหลับลึก')
                elif stress >= 7:
                    penalty += 1.0
                    reasons.append(f'**[ระดับความเครียดสูง]** สูงเกินเกณฑ์ปกติ ส่งผลเสียต่อคุณภาพการพักผ่อน')

                if disorder == 'Sleep Apnea':
                    penalty += 3.0
                    reasons.append('**[ความผิดปกติ]** พบสภาวะหยุดหายใจขณะหลับ (Sleep Apnea) ซึ่งอันตรายต่อระบบหัวใจ')
                elif disorder == 'Insomnia':
                    penalty += 1.5
                    reasons.append('**[ความผิดปกติ]** พบอาการนอนไม่หลับ ขัดขวางวงจรการนอนหลับปกติ')

                if bmi == 'Obese':
                    penalty += 0.8
                    reasons.append('**[ดัชนีมวลกาย]** ภาวะอ้วน (Obese) ส่งผลให้ทางเดินหายใจแคบลงขณะหลับ')
                if steps < 4000:
                    penalty += 0.5
                    reasons.append(f'**[กิจกรรม]** จำนวนก้าว ({steps}) น้อยเกินไป (Sedentary) ร่างกายจึงไม่พร้อมสำหรับการหลับลึก')

                final_score = max(0.0, min(10.0, raw_prediction - penalty))
                if final_score < 4.0:
                    status = 'วิกฤต'
                elif final_score < 6.0:
                    status = 'ควรปรับปรุงด่วน'
                elif final_score < 8.0:
                    status = 'พอใช้'
                else:
                    status = 'ดีเยี่ยม'

                st.divider()
                st.subheader(f'รายงานวิเคราะห์สุขภาพการนอน (วัย {age} ปี | เพศ {gender})')
                col_score, col_status = st.columns(2)
                with col_score:
                    st.metric('คะแนนประเมินสุทธิ', f'{final_score:.2f} / 10')
                with col_status:
                    st.metric('สถานะ', status)

                if reasons:
                    st.subheader('ปัจจัยที่เป็นอุปสรรคต่อคุณภาพการนอน')
                    for r in reasons:
                        st.markdown(f'- {r}')

                st.subheader('บทสรุปและคำแนะนำทางการแพทย์')
                if disorder == 'Sleep Apnea':
                    st.error('[ข้อควรระวังพิเศษ] ภาวะหยุดหายใจขณะหลับมีความเสี่ยงสูงต่อชีวิต ควรพบแพทย์ด่วนที่สุด')
                if stress >= 8:
                    st.info('ด้านสุขภาพจิต: ควรทำกิจกรรมผ่อนคลายก่อนนอนเพื่อลดระดับความเครียดสะสม')
                if steps < 7000:
                    st.info('ด้านกิจกรรม: การเดินให้ถึง 8,000 ก้าว จะช่วยให้ร่างกายหลับลึกได้ง่ายขึ้น')
                if bmi in ['Obese', 'Overweight']:
                    st.info('ด้านร่างกาย: การควบคุมน้ำหนักจะช่วยบรรเทาอาการกรนและหยุดหายใจขณะหลับได้')

            except Exception as e:
                st.error(f'เกิดข้อผิดพลาด: {e} (โปรดตรวจสอบรูปแบบ Blood Pressure เช่น 120/80)')
    
    with details_tab:
        st.markdown('### รายละเอียดของโมเดล Sleep Quality Prediction')
        st.info('เนื้อหาอธิบายโมเดลจะปรากฏที่นี่')


def run_flower_tab():
    st.subheader('Flower Classification')
    
    # Create subtabs for Usage and Details
    usage_tab, details_tab = st.tabs(['การใช้งานโมเดล', 'รายละเอียดของโมเดล'])
    
    with usage_tab:
        model = load_flower_model()
        class_names = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

        st.markdown('**ดอกไม้ที่ใช้ทำนาย 5 ชนิดได้แก่:**')
        st.write(f'🌼 {", ".join(class_names)})')
        st.write(f'สามารถดาวน์โหลดรูปภาพเพื่อใช้ทำนายโมเดลได้ที่นี่ https://drive.google.com/drive/folders/1z1lxFNUDpD2_af6pkcQyUwP7YEznUN9L?usp=drive_link')
        
        st.markdown('---')
        st.markdown('อัปโหลดรูปดอกไม้เพื่อทำนายประเภท')
        image_file = st.file_uploader('อัปโหลดภาพดอกไม้', type=['jpg', 'jpeg', 'png'])

        if image_file is not None:
            try:
                image = Image.open(image_file)
                st.image(image, caption='Uploaded Flower', use_column_width=True)

                if st.button('ทำนายประเภทดอกไม้', key='flower_predict'):
                    if model is None:
                        st.error('โมเดลดอกไม้ยังโหลดไม่สำเร็จ โปรดลองอีกครั้ง')
                        return

                    try:
                        # Preprocess the image
                        preprocessed = preprocess_flower_image(image)
                        
                        # Make prediction
                        predictions = model.predict(preprocessed, verbose=0)
                        confidence_scores = predictions[0]
                        predicted_class_idx = np.argmax(confidence_scores)
                        predicted_class = class_names[predicted_class_idx]
                        
                        # Display result
                        st.divider()
                        st.subheader('ผลการทำนาย')
                        st.metric('ประเภทดอกไม้', predicted_class)
                        
                    except Exception as e:
                        st.error(f'เกิดข้อผิดพลาดในการทำนาย: {str(e)}')

            except Exception as e:
                st.error(f'ไม่สามารถเปิดภาพได้: {str(e)[:150]}')
    
    with details_tab:
        st.markdown('### รายละเอียดของโมเดล Flower Classification')
        st.info('เนื้อหาอธิบายโมเดลจะปรากฏที่นี่')



# Main Tab Layout
st.title('Model Intelligent System')

tab1, tab2 = st.tabs(['Sleep Model', 'Flower Model'])

with tab1:
    run_sleep_tab()
with tab2:
    run_flower_tab()

