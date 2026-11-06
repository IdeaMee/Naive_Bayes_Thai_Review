# ระบบวิเคราะห์ความคิดเห็นจาก Wongnai-corpus

## ภาพรวมโครงการ

โครงการนี้เป็นระบบวิเคราะห์ความคิดเห็นจากรีวิวร้านอาหารภาษาไทยในชุดข้อมูล **Wongnai-corpus**  
โดยมีเป้าหมายจำแนกคะแนนรีวิวออกเป็น **5 ระดับดาว (1–5 ดาว)** ด้วยอัลกอริทึม **Naive Bayes**

ระบบใช้ **PyThaiNLP** สำหรับตัดคำภาษาไทย และใช้ **scikit-learn** สำหรับสร้างโมเดลจำแนกข้อความ  
ผลลัพธ์ประกอบด้วย Heatmap, Word Cloud และรายงานประสิทธิภาพของโมเดล

## ข้อมูลที่ใช้

- ชุดข้อมูล: Wongnai-corpus
- รูปแบบข้อมูล: รีวิวร้านอาหารภาษาไทย
- คอลัมน์ที่ใช้:
  - `review`: ข้อความรีวิว
  - `rating`: คะแนนรีวิว 1–5 ดาว
- ประเภทงาน: Text Classification แบบ 5 คลาส

## ขั้นตอนการทำงาน

### 1. ติดตั้งไลบรารี

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud pythainlp
python -m pythainlp.download thwiki
```

ไลบรารีหลักที่ใช้:

| ไลบรารี | หน้าที่ |
|---|---|
| pandas | อ่านและจัดการข้อมูล |
| PyThaiNLP | ตัดคำภาษาไทย |
| scikit-learn | สร้างโมเดล Naive Bayes และประเมินผล |
| matplotlib / seaborn | สร้าง Heatmap |
| wordcloud | สร้าง Word Cloud |

### 2. โหลดชุดข้อมูล

ระบบอ่านไฟล์ `w_review_train.csv` จาก Wongnai-corpus  
โดยไฟล์นี้แยกข้อมูลด้วยเครื่องหมาย `;` และไม่มีหัวตาราง

```python
df = pd.read_csv(
    "w_review_train.csv",
    sep=";",
    header=None,
    names=["review", "rating"]
)
```

จากนั้นเลือกเฉพาะรีวิวที่มีคะแนน 1–5 ดาว

### 3. ตรวจสอบและเตรียมข้อมูล

- ลบรีวิวซ้ำด้วย `drop_duplicates()`
- กรองคะแนนให้เหลือเฉพาะ 1, 2, 3, 4 และ 5
- แปลงคะแนนให้เป็นจำนวนเต็ม
- ตรวจสอบจำนวนรีวิวในแต่ละระดับดาว

การกระจายของคะแนนมีความสำคัญ เพราะหากคลาสใดมีข้อมูลน้อยมาก โมเดลอาจทำนายคลาสนั้นได้แม่นยำน้อยกว่าคลาสอื่น

### 4. ตัดคำภาษาไทยด้วย PyThaiNLP

ภาษาไทยไม่มีการเว้นวรรคระหว่างคำเหมือนภาษาอังกฤษ  
จึงต้องใช้ตัวตัดคำก่อนนำข้อความไปวิเคราะห์

ระบบใช้ `word_tokenize()` จาก PyThaiNLP ด้วยเอนจิน `newmm`

```python
from pythainlp.tokenize import word_tokenize

def thai_tokenize(text):
    tokens = word_tokenize(str(text), engine="newmm")
    tokens = [t.strip() for t in tokens if t.strip()]
    return " ".join(tokens)
```

ตัวอย่าง:

```text
ข้อความเดิม: อาหารอร่อยมากบริการดี
หลังตัดคำ: อาหาร อร่อย มาก บริการ ดี
```

### 5. แปลงข้อความเป็นตัวเลข

โมเดล Machine Learning ไม่สามารถอ่านข้อความโดยตรงได้  
จึงต้องแปลงข้อความให้เป็นเวกเตอร์ตัวเลข

ระบบใช้ `CountVectorizer` เพื่อนับความถี่ของคำในแต่ละรีวิว

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["tokens"])
```

- แต่ละคอลัมน์แทนคำหนึ่งคำ
- แต่ละแถวแทนรีวิวหนึ่งรายการ
- ค่าในตารางคือจำนวนครั้งที่คำนั้นปรากฏในรีวิว
- `max_features=5000` ใช้เพื่อจำกัดจำนวนคำ เพื่อให้โมเดลเร็วขึ้นและลดปัญหาข้อมูลกระจัดกระจาย

### 6. กำหนด Label

ระบบใช้คะแนนรีวิวเป็น label โดยตรง

```python
y = df["rating"]
```

คลาสที่ใช้จำแนกคือ:

| Label | ความหมาย |
|---|---|
| 1 | รีวิว 1 ดาว |
| 2 | รีวิว 2 ดาว |
| 3 | รีวิว 3 ดาว |
| 4 | รีวิว 4 ดาว |
| 5 | รีวิว 5 ดาว |

### 7. แบ่งข้อมูล Train และ Test

ข้อมูลถูกแบ่งเป็น 2 ส่วน:

- Train set: ใช้ฝึกโมเดล
- Test set: ใช้ทดสอบประสิทธิภาพโมเดลกับข้อมูลที่ไม่เคยเห็นมาก่อน

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

- `test_size=0.2` หมายถึงใช้ข้อมูล 20% ทดสอบ
- `random_state=42` ทำให้ผลการแบ่งข้อมูลซ้ำได้
- `stratify=y` ช่วยรักษาสัดส่วนของแต่ละระดับดาวใน train และ test ให้ใกล้เคียงกัน

### 8. สร้างและฝึกโมเดล Naive Bayes

ระบบใช้ `MultinomialNB` เหมาะกับข้อมูลที่เป็นจำนวนนับ เช่น ความถี่ของคำ

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

## หลักการทำงานของ Naive Bayes

Naive Bayes เป็นอัลกอริทึมการจำแนกประเภทที่ทำงานบนหลักของ **ทฤษฎีบท Bayes**

สูตรพื้นฐานคือ:

```text
P(class | words) = P(words | class) * P(class) / P(words)
```

ในบริบทของระบบนี้:

- `P(class | words)` คือความน่าจะเป็นที่รีวิวจะเป็นคะแนน d ดาว เมื่อพบคำในรีวิวนั้น
- `P(words | class)` คือความน่าจะเป็นที่จะพบคำเหล่านั้นในรีวิวระดับ d ดาว
- `P(class)` คือสัดส่วนของรีวิวแต่ละระดับดาวในชุดข้อมูล

โมเดลจะเลือกคะแนนที่มีค่าความน่าจะเป็นสูงที่สุดเป็นคำตอบ

### ความหมายของคำว่า Naive

คำว่า Naive หมายถึงโมเดลสมมติว่าคำแต่ละคำในรีวิวมีความเป็นอิสระต่อกัน

ตัวอย่างเช่น ในประโยค:

```text
อาหารอร่อยมาก บริการดี
```

โมเดลถือว่าคำว่า “อร่อย” และ “ดี” มีผลต่อการทำนายแยกจากกัน  
แม้ในความเป็นจริงคำเหล่านี้อาจมีความเกี่ยวข้องกันก็ตาม

แม้สมมติฐานนี้จะไม่สมบูรณ์ แต่ Naive Bayes ยังทำงานได้ดีและเร็วสำหรับงานจำแนกข้อความ

## การประเมินประสิทธิภาพ

### 1. Accuracy

Accuracy คือสัดส่วนของรีวิวที่โมเดลทำนายถูกต้อง

```text
Accuracy = จำนวนที่ทำนายถูก / จำนวนข้อมูลทั้งหมด
```

### 2. Precision

Precision บอกว่า จากรีวิวที่โมเดลทำนายว่าเป็น d ดาว มีกี่รายการที่เป็น d ดาวจริง

```text
Precision = TP / (TP + FP)
```

### 3. Recall

Recall บอกว่า จากรีวิวที่เป็น d ดาวจริง โมเดลจับได้กี่รายการ

```text
Recall = TP / (TP + FN)
```

### 4. F1-score

F1-score เป็นค่าเฉลี่ยแบบ harmonic mean ของ Precision และ Recall  
เหมาะใช้เมื่อข้อมูลแต่ละคลาสมีจำนวนไม่สมดุล

```text
F1-score = 2 * (Precision * Recall) / (Precision + Recall)
```

### 5. Classification Report

ระบบใช้ `classification_report()` เพื่อแสดง Precision, Recall และ F1-score แยกตามระดับดาว 1–5

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5]
))
```

## Heatmap: Confusion Matrix

Confusion Matrix ใช้ดูว่าโมเดลทำนายผิดในลักษณะใด

- แกนแนวตั้ง: คะแนนจริง
- แกนแนวนอน: คะแนนที่โมเดลทำนาย
- ช่องทแยงมุม: จำนวนที่ทำนายถูกต้อง
- ช่องอื่น: กรณีที่โมเดลทำนายผิด

ตัวอย่างเช่น ถ้าโมเดลทำนายรีวิว 4 ดาวเป็น 5 ดาวบ่อย จะเห็นค่าในช่อง “จริง 4 ดาว / ทำนาย 5 ดาว” สูง

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

labels = [1, 2, 3, 4, 5]
cm = confusion_matrix(y_test, y_pred, labels=labels)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
```

## Word Cloud

Word Cloud ใช้แสดงคำที่พบบ่อยในรีวิวแต่ละระดับดาว

- คำที่ปรากฏบ่อยจะแสดงใหญ่กว่า
- ระบบสร้าง Word Cloud แยกตามคะแนน 1–5 ดาว
- ช่วยให้เห็นแนวโน้มคำที่มักใช้ในรีวิวระดับต่าง ๆ

ตัวอย่างแนวโน้มที่อาจพบ:

| ระดับดาว | คำที่อาจพบบ่อย |
|---|---|
| 1 ดาว | แย่, ไม่อร่อย, รอนาน, แพง |
| 2 ดาว | ไม่ค่อย, ธรรมดา, ผิดหวัง |
| 3 ดาว | โอเค, ธรรมดา, พอใช้ |
| 4 ดาว | อร่อย, ดี, แนะนำ |
| 5 ดาว | อร่อยมาก, ประทับใจ, บริการดี, กลับมาอีก |

การสร้าง Word Cloud ภาษาไทยต้องระบุ `font_path` เป็นฟอนต์ไทย เช่น `THSarabunNew.ttf`  
มิฉะนั้นตัวอักษรไทยอาจแสดงเป็นสี่เหลี่ยม

```python
from wordcloud import WordCloud

wc = WordCloud(
    font_path="THSarabunNew.ttf",
    width=1000,
    height=600,
    background_color="white",
    max_words=100
).generate(text)
```

## การทำนายรีวิวใหม่

หลังฝึกโมเดลเสร็จ ระบบสามารถรับรีวิวใหม่และทำนายระดับดาวได้

ขั้นตอนคือ:

1. รับข้อความรีวิวใหม่
2. ตัดคำด้วย PyThaiNLP
3. แปลงคำเป็นเวกเตอร์ด้วย `CountVectorizer` เดิม
4. ส่งเข้าโมเดล Naive Bayes
5. ได้ผลทำนายเป็นคะแนน 1–5 ดาว
6. แสดงความน่าจะเป็นของแต่ละระดับดาว

```python
def predict_review(text):
    tokens = thai_tokenize(text)
    X_new = vectorizer.transform([tokens])
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]
    return pred, dict(zip(model.classes_, proba))
```

## ผลลัพธ์ที่ระบบสร้าง

| ผลลัพธ์ | คำอธิบาย |
|---|---|
| Accuracy | สัดส่วนการทำนายถูกต้องโดยรวม |
| Classification Report | Precision, Recall และ F1-score แยกตามระดับดาว |
| `confusion_matrix_heatmap_5star.png` | Heatmap แสดงผลการจำแนก 5 ระดับดาว |
| `wordcloud_1_star.png` ถึง `wordcloud_5_star.png` | Word Cloud แยกตามระดับดาว |
| ผลทำนายรีวิวใหม่ | ระดับดาวที่คาดการณ์และความน่าจะเป็นของแต่ละคลาส |

## ข้อจำกัดของระบบ

- Naive Bayes ถือว่าคำแต่ละคำเป็นอิสระต่อกัน จึงไม่เข้าใจลำดับคำหรือบริบทซับซ้อนอย่างสมบูรณ์
- การจำแนก 5 ระดับดาวยากกว่าการจำแนกเพียง positive และ negative เพราะความแตกต่างระหว่างรีวิว 3, 4 และ 5 ดาวอาจไม่ชัดเจน
- หากข้อมูลแต่ละระดับดาวไม่สมดุล โมเดลอาจมีแนวโน้มทำนายคลาสที่มีข้อมูลมากกว่าบ่อยขึ้น
- คำภาษาไทยบางคำอาจมีความหมายเปลี่ยนตามบริบท เช่น คำปฏิเสธ คำเสียดสี หรือคำแสลง

## แนวทางพัฒนาต่อ

- เพิ่มการลบ stop words ภาษาไทย
- ใช้ `TfidfVectorizer` แทน `CountVectorizer`
- ปรับค่า `alpha` ของ Multinomial Naive Bayes
- เพิ่มจำนวน `max_features` ของ CountVectorizer
- ใช้ TF-IDF ร่วมกับ Logistic Regression, SVM หรือ Random Forest
- พัฒนาต่อเป็นโมเดล Deep Learning เช่น LSTM หรือ Transformer สำหรับภาษาไทย
