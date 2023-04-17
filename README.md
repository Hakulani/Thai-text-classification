# Thai-text-classification
Thai-text-classification using transformer model
Thai Transformers using the Tcas61_2.csv dataset. As per your guidance, I have implemented various improvements to the model and am now submitting the final version.

Here is a summary of the fine-tuning process for the final model:
Summary fine tune model


Here is a summary of the fine-tuning process for the final model:

1.Balanced the dataset using RandomOverSampler:
Class distribution before RandomOverSampler:

0    0.653226
1    0.346774
Name: label, dtype: float64

Class distribution after RandomOverSampler:
0    0.5
1    0.5

Split train dataset into train, validation and test sets + 

Train 70% (Class 0:50% , Class 1:50%) 
 Val 15%   Test 15%


2.max_seq_len = 25

3. Chose the pre-trained WangchanBERTa model and froze its weights
bert = AutoModel.from_pretrained('poom-sci/WangchanBERTa-finetuned-sentiment')

4.batch_size = 32

5. Add more layer
# dropout layer
      self.dropout = nn.Dropout(0.1)

      # dense layer 1
      self.fc1 = nn.Linear(768,512)    
      # dense layer 2

      self.fc2 = nn.Linear(512,256)
      # dense layer 3 (Output layer)
      self.fc3 = nn.Linear(256,2)
 
6. define the optimizer and tuning learing rate
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

7.epochs = 100

8. Saved the best model with the lowest validation loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
Please find the following attachments for your review:

Google Colab notebook file with the final implemented Transformer model, training, and evaluation code
Screenshots of the captured results for both training and testing, including the F1 score
