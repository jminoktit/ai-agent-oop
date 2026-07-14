# Lightning AI Studio - خطوة بخطوة

## 1. حساب مجاني
- روح على https://lightning.ai/studio
- اضغط **"Get Started"** أو **"Sign Up"**
- سجّل بـ **Google** أو **GitHub**
- كده عندك **22 ساعة GPU مجانية** شهرياً

## 2. اعمل Studio جديد
- اضغط **"New Studio"** (الزرار الأزرق)
- اختار **"Start from scratch"**
- حط اسم: `AuraTrainer`
- اختار **GPU**: T4 (مجاني)

## 3. رفع الكود
### الطريقة الأولى: GitHub
- اضغط **"Clone from GitHub"**
- حط: `https://github.com/jminoktit/ai-agent-oop.git`
- اضغط **Clone**

### الطريقة الثانية: Upload
- اضغط **"Upload Files"**
- ارفع المجلد `AuraTrainer/scripts/lightning_train.py`

## 4. حط Environment Variables
- روح على **Settings** (ال Gear icon)
- اضغط **"Environment Variables"**
- حط_vars دي:
```
HF_TOKEN = your_hf_token_here
NOTIFY_EMAIL = your@email.com
SMTP_USER = your@gmail.com
SMTP_PASS = your_app_password
MODEL_NAME = google/gemma-2-2b-it
DATASET_SIZE = 100000
EPOCHS = 3
```

## 5. شغّل التدريب
- روح على **Terminal** (في Studio)
- اكتب:
```bash
cd ai-agent-oop
python AuraTrainer/scripts/lightning_train.py
```

## 6. شوف النتيجة
- التدريب هيشتغل ويبعت **logs** في Terminal
- لما يخلص، هيوصلك **email**
- الموديل هيتحفظ في `./checkpoints`

## 7. حمّل الموديل
- روح على **Files** (في Studio)
- دور على `./checkpoints`
- اضغط **Download**

## 8. لو عايز توقف
- في Terminal اضغط **Ctrl+C**

---

## ملاحظات مهمة:
- **GPU مجاني 22 ساعة/شهر** = تقريباً 3 جلسات تدريب
- **T4 GPU**: T4 = 16GB VRAM
- **الوقت**: ~50 دقيقة لكل 100K samples
- **السعر**: 0 (مجاني بالكامل)
