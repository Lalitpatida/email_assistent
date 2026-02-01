from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("IrisWiris/email-summarizer")
model = AutoModelForSeq2SeqLM.from_pretrained("IrisWiris/email-summarizer")
  
email = """
        Good morning,
    
        I hope this message finds you well. Unfortunately, I am unwell and will not be able to hold today's lecture or any classes for the remainder of this week. As a result, all classes are canceled.
        However, I uploaded a document that you should review before classes resume next week as scheduled. 
        Please ensure you go through the material, as it will be important for our upcoming discussions.
        
        Thank you for your understanding, and I look forward to seeing you next week.
    
        Best regards,
        James
  """
inputs = tokenizer(email, return_tensors="pt", truncation=True)
outputs = model.generate(inputs.input_ids, max_length=100)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
