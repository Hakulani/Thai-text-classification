 <head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
 </head>
<body>
    <h1>Thai-text-classification</h1>
    <p>Thai-text-classification using transformer model. Thai Transformers using the Tcas61_2.csv dataset.</p>
    <p>In this project, I developed a Thai text classification model using the state-of-the-art Transformer architecture.</p>
    <p>The primary goal was to achieve a high F1 score on the Tcas61_2.csv dataset, which contains Thai text data with binary labels.</p>
    <h2>Summary of the Fine-tuning Process</h2>
    <ol>
        <li>Balanced the dataset using RandomOverSampler:
            <ul>
                <li>Class distribution before RandomOverSampler:</li>
                <pre>
0    0.653226
1    0.346774
Name: label, dtype: float64
                </pre>
                <li>Class distribution after RandomOverSampler:</li>
                <pre>
0    0.5
1    0.5
                </pre>
                <li>Split the train dataset into train (70%: Class 0:50%, Class 1:50%), validation (15%), and test (15%) sets</li>
            </ul>
        </li>
        <li>Set the max_seq_len to 25</li>
        <li>Chose the pre-trained WangchanBERTa model and froze its weights:</li>
        <pre>
bert = AutoModel.from_pretrained('poom-sci/WangchanBERTa-finetuned-sentiment')
        </pre>
        <li>Set the batch_size to 32</li>
        <li>Added more layers to the model:
            <ul>
                <li>Dropout layer with a rate of 0.1</li>
                <li>Dense layer 1: self.fc1 = nn.Linear(768,512)</li>
                <li>Dense layer 2: self.fc2 = nn.Linear(512,256)</li>
                <li>Dense layer 3 (Output layer): self.fc3 = nn.Linear(256,2)</li>
            </ul>
        </li>
        <li>Defined the optimizer using AdamW with a learning rate of 2e-5 and weight_decay of 0.01</li>
        <li>Trained the model for 100 epochs</li>
        <li>Saved the best model with the lowest validation loss:</li>
        <pre>
if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'saved_weights.pt')
        </pre>
    </ol>

        
 <h2>Attachments</h2>
    <ul>
        <li>Google Colab notebook file with the final implemented Transformer model, training, and evaluation code</li>
        <li>Screenshots of the captured results for both training and testing, including the F1 score</li>
    </ul>
        
</body>
</html>
        by Witsarut Wongsim
