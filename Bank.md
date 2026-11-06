# AI Skill Definition: Thai_NLP_Naive_Bayes_Analyzer

## 1. Name & Description (ชื่อและคำอธิบาย)
- **Skill Name:** `Thai_NLP_Naive_Bayes_Analyzer`
- **Description:** ทักษะสำหรับสร้างระบบวิเคราะห์ความรู้สึก (Sentiment Analysis) สำหรับภาษาไทย โดยใช้เทคนิค Naive Bayes แบบเขียนเอง (From Scratch) ร่วมกับไลบรารีพื้นฐาน เช่น `numpy` และ `pythainlp` รวมถึงการทำ Data Visualization
- **Trigger conditions:** ควรเรียกใช้งาน Skill นี้เมื่อผู้ใช้มีคำสั่งเช่น:
  - "ช่วยเขียนโค้ดวิเคราะห์รีวิว Wongnai ให้หน่อย"
  - "ต้องการหาว่าคำไหนทำให้คะแนนดาวเพิ่มขึ้นโดยใช้ Naive Bayes"
  - "ทำ Word Cloud และ Heatmap จากข้อมูลภาษาไทย"

---

## 2. Guidelines / Rules & Constraints (ข้อกำหนดและสิ่งที่ห้ามทำ)

**กฎเหล็ก (Iron Rules):**
- ต้องใช้คณิตศาสตร์และโครงสร้างข้อมูลพื้นฐานของ Python (เช่น Dictionary, List) หรือ `numpy` ในการทำระบบนับคำ (Bag of Words) และสมการ Naive Bayes เท่านั้น
- ต้องจัดการกับสมการความน่าจะเป็นด้วย Log-probability เสมอ เพื่อป้องกันปัญหา Underflow (ค่าความน่าจะเป็นเข้าใกล้ 0 เกินไป)
- การทำเครื่องหมายทางคณิตศาสตร์ในคำอธิบายต้องใช้รูปแบบ LaTeX เสมอ (เช่น $P(C|X)$)

**Do’s (สิ่งที่ควรทำ):**
- **Type Hinting:** ให้ระบุ Data Type ของตัวแปรในฟังก์ชันเสมอ (เช่น `def predict(tokens: list[str]) -> int:`)
- **Code Comments:** อธิบายการทำงานของโค้ดแต่ละบล็อกเป็นภาษาไทยให้ชัดเจน
- **Smoothing:** ต้องทำ Laplace Smoothing (+1) ในขั้นตอนคำนวณ Likelihood เสมอ

**Don’ts (สิ่งที่ไม่ควรทำ):**
- **ห้าม** ใช้ไลบรารี Machine Learning สำเร็จรูป เช่น `scikit-learn` (ห้ามใช้ `MultinomialNB` หรือ `CountVectorizer`) เว้นแต่จะระบุไว้เป็นอย่างอื่น
- **ห้าม** ข้ามขั้นตอนการกรอง Stop words หรือตัวอักษรขยะภาษาไทย
- **ห้าม** แทรก Inline Style หรือ CSS หากมีการจำลองผลลัพธ์เป็น HTML (ให้ใช้โครงสร้าง Markdown หรือตัวแปรแทน)

---

## 3. Step-by-Step Workflow (ลำดับขั้นตอนการทำงาน)
เวลา AI ได้รับงานนี้ จะต้องทำงานตามลำดับดังต่อไปนี้อย่างเคร่งครัด:

- **Step 1: Data Initialization & Preprocessing**
  - นำเข้าไลบรารีที่ได้รับอนุญาต (`numpy`, `pandas`, `pythainlp`, `matplotlib`, `wordcloud`)
  - สร้างฟังก์ชันสำหรับทำความสะอาดข้อความและตัดคำ (Tokenization) โดยกรอง Stop words ออก
- **Step 2: Dictionary / Bag of Words Construction**
  - วนลูปอ่านข้อมูลเพื่อสร้าง Vocabulary (คำศัพท์ทั้งหมด)
  - นับความถี่ของคำในแต่ละคลาส (Class Frequencies)
- **Step 3: Model Training (Probability Calculation)**
  - คำนวณ Prior Probability $P(Class)$ ของทุกคลาส
  - คำนวณ Likelihood $P(Word \vert Class)$ โดยประยุกต์ใช้ Laplace Smoothing
- **Step 4: Evaluation & Visualization**
  - สร้างฟังก์ชัน `predict()` สำหรับทดสอบชุดข้อมูล Test set
  - สร้าง Confusion Matrix และแสดงผลในรูปแบบ Heatmap
- **Step 5: Feature Importance Extraction**
  - คำนวณหาอัตราส่วนเพื่อหา "คำที่มีอิทธิพลเชิงบวกสูงที่สุด" (เช่น นำค่าความน่าจะเป็นของ 5 ดาว หารด้วย 1 ดาว)
  - นำผลลัพธ์ที่ได้ไปสร้าง Word Cloud เพื่อแสดงภาพให้ชัดเจน

---

## 4. Examples (ตัวอย่างโค้ดหรือผลลัพธ์ต้นแบบ)

เพื่อให้เห็น Pattern การเขียนโค้ดที่ถูกต้องตามกฎ (มีการใช้ Type Hint, ไม่ใช้ ML Library สำเร็จรูป, และคอมเมนต์ภาษาไทย):

```python
import math
from typing import List, Dict

# ตัวอย่าง Step 3: การคำนวณ Likelihood พร้อม Type Hinting
def calculate_likelihoods(word_counts: Dict[int, Dict[str, int]], 
                          vocab_size: int, 
                          classes: List[int]) -> Dict[int, Dict[str, float]]:
    """
    ฟังก์ชันคำนวณ Likelihood โดยใช้ Laplace Smoothing แบบเขียนเอง
    """
    log_likelihoods: Dict[int, Dict[str, float]] = {}
    
    for c in classes:
        log_likelihoods[c] = {}
        # นับจำนวนคำทั้งหมดที่โผล่มาในคลาส c
        total_words_in_class = sum(word_counts[c].values())
        
        for word in vocab:
            # ใช้ Laplace Smoothing: +1 ที่เศษ และ +vocab_size ที่ส่วน
            count = word_counts[c].get(word, 0)
            prob = (count + 1) / (total_words_in_class + vocab_size)
            # เก็บค่าเป็น Log เพื่อป้องกัน Underflow
            log_likelihoods[c][word] = math.log(prob)
            
    return log_likelihoods
```