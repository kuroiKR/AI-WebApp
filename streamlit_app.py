import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Model Intelligent System",
    page_icon="🍀",
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
        st.markdown("""
## แนวทางการพัฒนาโมเดลคาดการณ์คุณภาพการนอน (Sleep Quality Prediction)
                    
**การเตรียมข้อมูล (Data Preparation)**

โปรเจกต์นี้เริ่มต้นด้วยการรวบรวมข้อมูลสุขภาพการนอนผ่านการสืบค้นโดย ChatGPT เพื่อระบุแหล่งข้อมูลที่มีความน่าเชื่อถือสูงจากสถาบันระดับโลก ได้แก่ Kaggle (Sleep Health and Lifestyle Dataset), American Academy of Sleep Medicine (AASM), National Sleep Foundation (NSF) และข้อมูลสำรวจจาก CDC ลักษณะของข้อมูล:

Dataset ประกอบด้วยตัวแปรสำคัญ 11 ตัวแปร ที่ส่งผลต่อประสิทธิภาพการนอน ได้แก่:

1. ข้อมูลพื้นฐาน: เพศ (Gender), อายุ (Age), อาชีพ (Occupation)

2. พฤติกรรมสุขภาพ: ระยะเวลาการนอน (Sleep Duration), คุณภาพการนอน (Quality of Sleep), ระดับกิจกรรมทางกาย (Physical Activity Level), ระดับความเครียด (Stress Level) , ความผิดปกติของการนอน (Sleep Disorder)
3. ตัวชี้วัดทางกายภาพ: ประเภทดัชนีมวลกาย (BMI Category), ความดันโลหิต (Blood Pressure), อัตราการเต้นของหัวใจ (Heart Rate), จำนวนก้าวต่อวัน (Daily Steps)

ขั้นตอนการประมวลผล:

- Data Cleaning: จัดการข้อมูลที่สูญหาย (Missing Values) และตรวจสอบความสมเหตุสมผลของค่าความดันโลหิต

- Feature Engineering: แปลงข้อมูลเชิงคุณภาพ (เช่น BMI Category) ให้เป็นตัวเลข และทำ Feature Scaling เพื่อให้โมเดลประมวลผลข้อมูลที่มีหน่วยต่างกันได้อย่างแม่นยำ

- Data Splitting: แบ่งข้อมูลเป็น Training Set 80% และ Test Set 20% เพื่อใช้ประเมินประสิทธิภาพที่แท้จริง

## ทฤษฎีของอัลกอริทึม (Algorithm Theory)

เพื่อให้ได้ผลลัพธ์ที่แม่นยำที่สุด โปรเจกต์นี้เลือกใช้เทคนิค Ensemble Voting Regressor ซึ่งเป็นการรวมพลังของ 4 อัลกอริทึมหลัก:

1. Random Forest

ใช้หลักการ Bagging สร้างต้นไม้ตัดสินใจ (Decision Trees) จำนวนมากแบบสุ่ม แล้วนำค่าเฉลี่ยมาเป็นคำตอบ ช่วยลดปัญหาการจดจำข้อมูลแม่นเกินไป (Overfitting) ได้ดีเยี่ยม

2. Gradient Boosting

ใช้หลักการ Boosting ที่สร้างต้นไม้ทีละต้น โดยต้นที่มาทีหลังจะพยายามเรียนรู้และ "แก้ไขความผิดพลาด" (Residual Errors) จากต้นก่อนหน้า ทำให้โมเดลมีความละเอียดสูง

3. Support Vector Regression (SVR)

ใช้ทฤษฎี Hyperplane และ Kernel Trick เพื่อหาเส้นแนวโน้มในพื้นที่หลายมิติ เหมาะสำหรับข้อมูลที่มีความสัมพันธ์ที่ซับซ้อนและไม่ใช่เส้นตรง

4. XGBoost

เป็นเวอร์ชันปรับปรุงของ Gradient Boosting ที่เน้นความเร็วและความแม่นยำสูง มีระบบ Regularization ช่วยป้องกันโมเดลไม่ให้ซับซ้อนเกินไปจนสูญเสียความแม่นยำกับข้อมูลใหม่

5. Ensemble Voting

ทำหน้าที่เป็น "คณะกรรมการ" รวบรวมคำทำนายจากทุกโมเดลข้างต้นมาหาค่าเฉลี่ยถ่วงน้ำหนัก เพื่อลดความลำเอียง (Bias) และเพิ่มความเสถียรของผลลัพธ์

## ขั้นตอนการพัฒนาโมเดล (Development Process)

การพัฒนาแบ่งออกเป็น 7 ระยะ (Phases) เพื่อความเป็นระบบ:

1. Data Collection & Preparation: รวบรวมและคัดเลือก Dataset ที่เกี่ยวข้องกับสุขภาพการนอน

2. Exploratory Data Analysis (EDA): วิเคราะห์ความสัมพันธ์ของตัวแปร เช่น ความสัมพันธ์ระหว่าง "ระดับความเครียด" กับ "คุณภาพการนอน" ผ่าน Heatmap

3. Feature Engineering & Preprocessing: ปรับแต่งข้อมูลดิบให้พร้อมสำหรับอัลกอริทึม (Scaling, Encoding)

4. Model Training & Selection: ฝึกสอนโมเดลทั้ง 4 ประเภท และเปรียบเทียบประสิทธิภาพเบื้องต้น

5. Hyperparameter Tuning & Optimization: ปรับแต่งค่าพารามิเตอร์ภายในของแต่ละโมเดล (เช่น จำนวนต้นไม้ใน Forest) เพื่อให้ได้จุดที่แม่นยำที่สุด

6. Model Evaluation & Validation: วัดผลด้วยตัวชี้วัด MAE (Mean Absolute Error) และ $R^2$ Score เพื่อดูว่าโมเดลอธิบายความผันแปรของข้อมูลได้ดีเพียงใด

7. Deployment & Testing: บันทึกโมเดลในรูปแบบไฟล์ .joblib เพื่อนำไปประกอบใช้งานใน Interface ต่อไป

## แหล่งอ้างอิงข้อมูลที่นำมาใช้ (References)

- Datasets: * Kaggle: Sleep Health and Lifestyle Dataset  https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

- National Health and Nutrition Examination Survey (NHANES) - CDC

- Literature & Theory: * American Academy of Sleep Medicine (AASM) Clinical Practice Guidelines

- Scikit-learn Documentation (Random Forest, SVR, Ensemble)

- XGBoost Documentation: Scalable Tree Boosting System

## วิธีการใช้งานโมเดล (How to Use)

เมื่อเปิดใช้งานโมเดล ให้ไปที่ส่วนการกรอกข้อมูลสุขภาพเพื่อรับการประเมิน ดังนี้:

1. Age -> อายุของผู้ใช้งาน (ปี)

2. Gender -> เพศสภาพของผู้ใช้งาน

3. Occupation -> อาชีพของผู้ใช้งาน

4. Sleep Duration -> จำนวนชั่วโมงที่นอนจริง (เช่น 7.5 ชั่วโมง)

5. Physical Activity Level -> ปริมาณการเคลื่อนไหวร่างกายในแต่ละวัน

6. Stress Level -> ระดับความเครียดประเมินตนเอง 1–10

7. BMI Category -> สถานะน้ำหนัก (Normal, Overweight, Obese)
                    
8. Blood Pressure -> ความดันโลหิตแบบ Systolic/Diastolic (เช่น 120/80 mmHg)

9. Heart Rate -> อัตราการเต้นของหัวใจขณะพัก (BPM)

10. Daily Steps -> จำนวนก้าวที่เดินเฉลี่ยต่อวัน

11. Sleep Disorder -> ภาวะผิดปกติของการนอนหลับ คือ สภาวะที่ส่งผลกระทบต่อคุณภาพ ระยะเวลา และช่วงเวลา ของการนอนหลับ

ꕤ  ขั้นตอนการใช้งาน:

1. Input Data: กรอกข้อมูลสุขภาพตามความเป็นจริงในช่องที่กำหนด

2. Processing: ระบบจะนำข้อมูลไปปรับสเกลให้เข้ากับมาตรฐานที่โมเดลเรียนรู้มา

3. Prediction: กดปุ่ม "ประเมินสุขภาพการนอน" โมเดล Ensemble จะคำนวณผลลัพธ์

4. Output: ระบบจะแสดง คะแนนคุณภาพการนอน พร้อมคำแนะนำเบื้องต้นตามผลวิเคราะห์

หมายเหตุ: โมเดลนี้เป็นการคาดการณ์ทางสถิติเพื่อการดูแลสุขภาพเบื้องต้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์จากผู้เชี่ยวชาญหรือผลตรวจ Sleep Test ได้

                    

                    
ผู้พัฒนา Project : นางสาวณัฐณิชา ถนอมลาภ , นางสาวณิรชา ถนอมลาภ
""")

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
        st.markdown("""
## แนวทางการพัฒนาโมเดลจำแนกชนิดดอกไม้ (Flower Classification)
** การเตรียมข้อมูล (Data Preparation) **

โปรเจกต์นี้ใช้ชุดข้อมูลภาพดอกไม้ที่มีความหลากหลายสูง เพื่อให้โมเดลสามารถจดจำลักษณะเด่นของดอกไม้แต่ละชนิดได้อย่างแม่นยำ โดยมีกระบวนการจัดการข้อมูลดังนี้:

- แหล่งที่มาของข้อมูล: ใช้ Dataset จาก Kaggle: Flowers Dataset ซึ่งประกอบด้วยภาพดอกไม้ 5 ประเภทหลัก ได้แก่ ดอกเดซี่ (Daisy), ดอกแดนดิไลอัน (Dandelion), ดอกกุหลาบ (Rose), ดอกทานตะวัน (Sunflower) และดอกทิวลิป (Tulip)

- การคัดกรองข้อมูล (Data Cleaning): ดำเนินการตรวจสอบและคัดแยกรูปภาพที่ไม่เกี่ยวข้อง (Outliers) หรือภาพที่มีความผิดเพี้ยนสูงออก เพื่อป้องกันโมเดลเกิดความสับสนในการเรียนรู้ลักษณะทางกายภาพที่แท้จริงของดอกไม้

- การประมวลผลภาพ (Image Preprocessing):

  - Resizing: ปรับขนาดภาพทุกภาพให้เป็นขนาดมาตรฐาน (เช่น $224 \\times 224$ หรือ $128 \\times 128$ พิกเซล) เพื่อให้เข้าสู่โครงข่ายประสาทเทียมได้

  - Normalization: ปรับค่าพิกเซลของภาพให้อยู่ในช่วง 0 ถึง 1 เพื่อช่วยให้โมเดลประมวลผลได้รวดเร็วและมีเสถียรภาพมากขึ้น

  - Data Augmentation: เพิ่มความหลากหลายของข้อมูลด้วยการ หมุนภาพ (Rotation), การพลิกภาพ (Flipping) และการปรับความสว่าง เพื่อให้โมเดลมีความทนทาน (Robustness) ต่อภาพถ่ายในสภาพแวดล้อมที่แตกต่างกัน

- การแบ่งข้อมูล: แบ่งชุดข้อมูลออกเป็น Training Set (80%) สำหรับการฝึกสอน และ Test Set (20%) สำหรับการทดสอบประสิทธิภาพ

** ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory) **

โมเดลนี้พัฒนาโดยใช้พื้นฐานของ Artificial Neural Network (ANN) โดยเน้นสถาปัตยกรรมแบบ Convolutional Neural Network (CNN) ซึ่งเป็นโครงข่ายประสาทเทียมที่ออกแบบมาเพื่อประมวลผลข้อมูลภาพโดยเฉพาะ ประกอบด้วยกลไกสำคัญดังนี้:

1. Convolutional Layer: ทำหน้าที่เป็นตัวสกัดฟีเจอร์ (Feature Extraction) โดยการใช้ Filter เล็กๆ กวาดไปบนภาพเพื่อตรวจจับเส้น ขอบ รูปร่าง และลวดลายของกลีบดอกไม้หรือเกสร

2. Pooling Layer (Max Pooling): ทำหน้าที่ลดขนาดของข้อมูล (Downsampling) โดยเลือกดึงเฉพาะค่าที่เด่นที่สุดในพื้นที่นั้นๆ ช่วยลดภาระการคำนวณและป้องกันการ Overfitting

3. Activation Function (ReLU): ใช้ฟังก์ชัน ReLU (Rectified Linear Unit) เพื่อเพิ่มความเป็น Non-linear ให้กับโมเดล ทำให้สามารถเรียนรู้แพทเทิร์นที่ซับซ้อนของรูปทรงดอกไม้ได้ดีขึ้น

4. Fully Connected Layer (Dense Layer): ส่วนสุดท้ายของโครงข่ายที่นำฟีเจอร์ทั้งหมดที่สกัดได้มาประมวลผลรวมกันเพื่อตัดสินใจ

5. Softmax Function: ใช้ใน Layer สุดท้ายเพื่อแปลงผลลัพธ์ให้ออกมาเป็นค่า "ความน่าจะเป็น" (Probability) ของดอกไม้แต่ละชนิด โดยผลรวมของทุกคลาสจะเท่ากับ 1 (หรือ 100%)

** ขั้นตอนการพัฒนาโมเดล (Development Process) **

กระบวนการพัฒนาถูกดำเนินการอย่างเป็นขั้นตอนเพื่อให้ได้ความแม่นยำสูงสุด:

1. Environment Setup: ตั้งค่าสภาพแวดล้อมการทำงานด้วย Python ร่วมกับ Library หลักอย่าง TensorFlow, Keras และ OpenCV

2. Dataset Loading & Labeling: โหลดข้อมูลภาพและทำการทำ Label ข้อมูลตามโฟลเดอร์ประเภทดอกไม้

3. Model Architecture Design: ออกแบบโครงสร้าง Layer ของ CNN โดยวางลำดับจาก Conv2D, MaxPooling ไปจนถึง Dense Layer

4. Model Compilation: กำหนด Loss Function แบบ Categorical Crossentropy และใช้ Optimizer แบบ Adam ซึ่งมีประสิทธิภาพสูงในการปรับปรุงน้ำหนัก (Weights) ของโมเดล

5. Model Training: ทำการ Train โมเดลผ่านรอบการเรียนรู้ (Epochs) และตรวจสอบค่า Accuracy และ Loss อย่างต่อเนื่อง

6. Evaluation: ประเมินโมเดลด้วย Confusion Matrix เพื่อดูว่าโมเดลจำแนกดอกไม้ชนิดใดสับสนกับชนิดใดบ้าง

7. Deployment: บันทึกโมเดลในรูปแบบไฟล์ .h5 หรือรูปแบบที่เหมาะสม เพื่อนำไปเชื่อมต่อกับ Web Application (Streamlit)

** แหล่งอ้างอิงข้อมูลที่นำมาใช้ (References) **

- Dataset Source: Kaggle - Flowers Dataset by Rahma Sleam https://www.kaggle.com/datasets/rahmasleam/flowers-dataset

** วิธีการใช้งานโมเดล (How to Use) **

ผู้ใช้งานสามารถจำแนกชนิดของดอกไม้ได้ง่ายๆ ผ่านหน้าจอ Interface ดังนี้:

1. Browse files -> อัปโหลดรูปภาพดอกไม้ที่ต้องการทราบชื่อ (รองรับไฟล์ .jpg, .png, .jpeg)

2. Uploaded Flower -> ระบบจะแสดงรูปภาพที่อัปโหลดเพื่อให้ผู้ใช้ตรวจสอบความถูกต้อง

3. ทำนายประเภทดอกไม้ -> ปุ่มสำหรับสั่งการให้โมเดล Neural Network เริ่มการประมวลผล

4. ผลการทำนาย -> แสดงชื่อประเภทดอกไม้ที่ตรวจพบ (เช่น "Rose" หรือ "Sunflower")

** ꕤ  ขั้นตอนการใช้งาน: **

1. Prepare Photo: เตรียมรูปภาพดอกไม้ที่เห็นลักษณะกลีบและเกสรชัดเจน

2. Upload: ลากและวางไฟล์ภาพลงในช่องอัปโหลดบนหน้าเว็บไซต์

3. Process: ระบบจะส่งข้อมูลรูปภาพส่งผ่านเข้าไปใน Neural Network ที่ถูกฝึกสอนมาแล้ว

4. Output: หน้าจอจะแสดงผลลัพธ์ชื่อดอกไม้ที่ได้ทำการทำนายจากรูปภาพออกมาอย่างแม่นยำ

หมายเหตุ: เพื่อความแม่นยำสูงสุด ควรใช้ภาพที่มีแสงสว่างเพียงพอและตัวดอกไม้อยู่บริเวณกึ่งกลางภาพ

                    

                    

ผู้พัฒนา Project : นางสาวณัฐณิชา ถนอมลาภ , นางสาวณิรชา ถนอมลาภ
""")


# Main Tab Layout
st.title('Model Intelligent System')

tab1, tab2 = st.tabs(['Sleep Model', 'Flower Model'])

with tab1:
    run_sleep_tab()
with tab2:
    run_flower_tab()

