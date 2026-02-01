from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
model = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")

# Sample text
texts = [
    # Spam examples
    "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now.",
    "URGENT: Your bank account has been compromised. Verify your details immediately.",
    "Limited time offer!!! Get 90% off on premium courses. Buy now!",
    "You have been selected for a free iPhone 15. Click the link to confirm.",
    "Earn $5000 per week working from home. No experience required.",
    "Dear user, your PayPal account will be suspended unless you act now.",
    "Exclusive deal just for you! Lowest price guaranteed. Shop today.",

    # Non-spam (ham) examples
    "Hey, are we still meeting for lunch today?",
    "Please find the attached report for last quarter’s performance.",
    "The meeting has been rescheduled to 3 PM tomorrow.",
    "Can you review my PR when you get a chance?",
    "Happy birthday! Hope you have a great year ahead 🎉",
    "I’ll be working from home today due to network issues.",
    "Your order has been shipped and will arrive by Friday."
]


# Tokenize the input
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=1)

# Get predicted labels
predictions = torch.argmax(probabilities, dim=1)

# Map labels to class names
label_map = {0: "Not Spam", 1: "Spam"}
for text, prediction in zip(texts, predictions):
    print(f"Text: {text}\nPrediction: {label_map[prediction.item()]}\n")








# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model_name = "Goodmotion/spam-mail-classifier"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# texts = [
# 'Join us for a webinar on AI innovations',
# 'Urgent: Verify your account immediately.',
# 'Meeting rescheduled to 3 PM',
# 'Happy Birthday!',
# 'Limited time offer: Act now!',
# 'Join us for a webinar on AI innovations',
# 'Claim your free prize now!',
# 'You have unclaimed rewards waiting!',
# 'Weekly newsletter from Tech World',
# 'Update on the project status',
# 'Lunch tomorrow at 12:30?',
# 'Get rich quick with this amazing opportunity!',
# 'Invoice for your recent purchase',
# 'Don\'t forget: Gym session at 6 AM',
# 'Join us for a webinar on AI innovations',
# 'bonjour comment allez vous ?',
# 'Documents suite à notre rendez-vous',
# 'Valentin Dupond mentioned you in a comment',
# 'Bolt x Supabase = 🤯',
# 'Modification site web de la société',
# 'Image de mise en avant sur les articles',
# 'Bring new visitors to your site',
# 'Le Cloud Éthique sans bullshit',
# 'Remix Newsletter #25: React Router v7',
# 'Votre essai auprès de X va bientôt prendre fin',
# 'Introducing a Google Docs integration, styles and more in Claude.ai',
# 'Carte de crédit sur le point d’expirer sur Cloudflare'
# ]
# inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
# outputs = model(**inputs)

# logits = outputs.logits
# probabilities = torch.softmax(logits, dim=1)

# labels = ["NOSPAM", "SPAM"] 
# results = [
#     {"text": text, "label": labels[torch.argmax(prob).item()], "confidence": prob.max().item()}
#     for text, prob in zip(texts, probabilities)
# ]

# for result in results:
#     print(f"Texte : {result['text']}")
#     print(f"Résultat : {result['label']} (Confiance : {result['confidence']:.2%})\n")
