### 4.1 ตารางเปรียบเทียบภาพรวม (Summary Mapping Table)

| ขั้นตอน  | ความหมายทางทฤษฎี | แนวทางสำหรับชุดข้อมูลรีวิว Wongnai |
| :--- | :--- | :--- |
| **1. Get training set จาก Dataset Wongnai** | ดึงข้อมูลข้อความรีวิว และกำหนดเป้าหมาย (Target) | • ข้อความรีวิว (Review Comment) = Input Instance ($Doc$)<br>• จำนวนดาว (1, 2, 3, 4, 5 ดาว) หรือกลุ่มความรู้สึก (Positive/Negative) = Target Class ($v_j$) |
| **2. ตัด Stop Word ออก โดยใช้ PyThaiNLP** | ลดสัญญาณรบกวน (Noise) และตัดคำที่ไม่ช่วยในการจำแนก | ใช้รายการคำหยุดภาษาไทยจาก `pythainlp.corpus.thai_stopwords()` เช่น "และ", "ก็", "ที่", "เป็น" |
| **3. Stemming / Tokenization โดยใช้ PyThaiNLP** | ตัดคำและจัดกลุ่มรูปคำให้อยู่ในรูปแบบมาตรฐานเดียวกัน | ใช้ `pythainlp.tokenize.word_tokenize` เพื่อตัดประโยคภาษาไทยเป็นโทเค็นคำ (Token) เนื่องจากภาษาไทยไม่มีเว้นวรรคระหว่างคำ |
| **4. ผ่าน Smoothing และค่าจะไม่เป็น 0 เมื่อคูณ** | ป้องกัน Zero-Frequency Problem ($m$-estimate จาก Slide 26) | ใช้ Laplace Smoothing ($+1$) ในการคำนวณ $P(w_k \| \text{rating})$ เพื่อไม่ให้คำที่ไม่เคยพบในบางกลุ่มดาวทำให้ผลคูณความน่าจะเป็นรวมกลายเป็น $0$ |
| **5. ตารางที่คำนวณได้จะถูกนำไปเก็บใน Memory** | จัดเก็บ Conditional Probability Table สำหรับทำ Lookup ตอน Inference | นำตารางความน่าจะเป็นของคำต่อคลาส (Conditional Probability Matrix) เก็บลงตัวแปร NumPy อาร์เรย์ในหน่วยความจำ เพื่อดึงค่ามาคำนวณได้อย่างรวดเร็ว |
| **6. Visualization & Performance Test** | สรุปพฤติกรรมข้อมูลและวัดประสิทธิภาพโมเดล | • **Word Cloud:** สรุปคำเด่นในภาพรวม และแยกรายกลุ่มดาว 1, 2, 3, 4, 5<br>• **Heatmap:** แสดงตารางความน่าจะเป็นของคำสำคัญ หรือ Confusion Matrix ด้วย Matplotlib<br>• **Accuracy:** วัดความแม่นยำบน Test Set |
| **จุดประสงค์: อยากรู้ว่า Naive Bayes ทำงานได้ดีในงานนี้มั้ย** | ประเมินความเหมาะสมของสมมติฐาน Naive Bayes กับงาน NLP จริง | พิสูจน์ว่าสมมติฐาน Conditional Independence (Slide 22) สามารถจำแนกความรู้สึกจากรีวิวร้านอาหารภาษาไทยได้มีประสิทธิภาพหรือไม่ |

---
