#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --user tensorflow==2.15')



# In[1]:


import pandas as pd
# Load DataFrame
df = pd.read_csv('youtube_comments_cleaned.csv')  # Replace with your actual dataset file

# Preview
print(df.head())


# In[2]:


print(df.columns)


# In[3]:


df = pd.read_csv('youtube_comments_cleaned.csv')  # Replace with your actual CSV file

# Check available columns
print("Columns before removing labels:", df.columns)

# -----------------------------
# Remove existing label columns if any
# -----------------------------
for col in ['label', 'label_num']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"✅ Removed existing '{col}' column.")

print("Columns after removal:", df.columns)

# -----------------------------
# Define toxic keywords
# -----------------------------
toxic_keywords = [ # English curses
    'fuck', 'fucking', 'fucked', 'shit', 'shitty', 'bullshit', 'bitch', 'bitches',
    'ass', 'asshole', 'motherfucker', 'mf', 'cunt', 'dick', 'dicks', 'cock',
    'pussy', 'faggot', 'fag', 'dyke', 'tranny', 'nigger', 'nigga', 'chink', 'spic', 'kike',
    'slut', 'whore', 'bastard', 'retard', 'retarded', 'moron', 'idiot', 'stupid', 'dumb',
    'loser', 'worthless', 'pathetic', 'disgusting', 'fat', 'ugly', 'kill yourself', 'kys',
    'die', 'drop dead', 'kill', 'burn in hell', 'go to hell', 'hate you', 'villain', 'menace',
    'cum', 'cumming', 'jerk off', 'jerk', 'witch', 'demon'
    
]
# Ensure no NaN and all string type
df = df.dropna(subset=['comment'])  # Remove rows with NaN comments
df['comment'] = df['comment'].astype(str)  # Convert all comments to strings

# -----------------------------
# Create new labels based on toxic keywords
# -----------------------------
def assign_label(comment):
    comment = str(comment).lower()
    for word in toxic_keywords:
        if word in comment:
            return 1  # toxic
    return 0  # non-toxic

# Apply labeling function
df['label_num'] = df['comment'].apply(assign_label)

print("✅ New labels created based on toxic keywords.")
print(df[['comment', 'label_num']].head(10))

# -----------------------------
# Save updated DataFrame with new labels
# -----------------------------
df.to_csv('youtube_comments_with_new_labels.csv', index=False)
print("✅ Saved updated dataset with new labels.")



# In[4]:


from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import class_weight



# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['comment'].tolist(),
    df['label_num'].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Prepare TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(1000).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
)).batch(16)

# Compute class weights
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_tensor = tf.constant(class_weights_array, dtype=tf.float32)

# Define weighted loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def weighted_loss(y_true, y_pred):
    weights = tf.gather(class_weights_tensor, tf.cast(y_true, tf.int32))
    unweighted_loss = loss_object(y_true, y_pred)
    weighted_loss = unweighted_loss * weights
    return tf.reduce_mean(weighted_loss)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=weighted_loss, metrics=['accuracy'])

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    callbacks=[early_stop]
)

# Save model
model.save_pretrained('toxic_classifier_model')
tokenizer.save_pretrained('toxic_classifier_model')


# In[9]:


# Final hybrid prediction pipeline
# -----------------------------
from transformers import pipeline

classifier = pipeline("text-classification", model="toxic_classifier_model", tokenizer="toxic_classifier_model", framework="tf")

def predict_toxicity(comment):
   comment_lower = comment.lower()
   for word in toxic_keywords:
       if word in comment_lower:
           return {'label': 'TOXIC', 'score': 1.0}  # keyword override
   
   # Else use model prediction
   result = classifier(comment)[0]
   label = 'TOXIC' if result['label'] == 'LABEL_1' else 'NON-TOXIC'
   return {'label': label, 'score': result['score']}


# In[10]:


# Test predictions
print(predict_toxicity("fuck"))
print(predict_toxicity("you are amazing"))
print(predict_toxicity("shut up idiot"))
print(predict_toxicity("go to hell moron"))


# In[ ]:




