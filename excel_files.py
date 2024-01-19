import pandas as pd
import openai


# Replace 'your_file.xlsx' with the actual path to your Excel file
file_path = 'macys_master_device_list.xlsx'

# Read the Excel file into a pandas DataFrame
excel_content = pd.read_excel(file_path)

# Define a question about the Excel file
user_question = "What is the IP of Docker 1?"

# Combine the prompt and the question
prompt = f"Excel File:\n{excel_content}\nUser Question: {user_question}\nAnswer:"

# Use GPT-3.5 to generate a response
response = openai.Completion.create(
  engine="text-davinci-002",  # You can choose a different engine based on your needs
  prompt=prompt,
  temperature=0.7,  # Adjust temperature for response randomness
  max_tokens=150  # Adjust max_tokens for response length
)

# Extract the generated answer from the response
generated_answer = response['choices'][0]['text'].strip()

# Display the generated answer
print("Generated Answer:", generated_answer)
