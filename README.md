# Train GPT2 model in pytorch

## To train a GPT 2 model, follow the below steps.

<b>1. Prepare data</b>
<br>Use the huggingface bookcorpusopen dataset as an example, run the preparedata.py to tokenize the dataset and store in disk, each record will include 513 token id in contiguous.

<b>2. Train the model</b>
<br>Run the train.py

<b>3. Generate text</b>
<br>Run the predict.py

## To train the ChatGPT use the SFT, follow the below steps.

<b>1. Prepare the prompt and response data</b>
<br>Run the preparechatdata.py

<b>2. Finetune the GPT 2 model</b>
<br>Run the prompt_train.py

<b>3. Chat with the model</b>
<br>Run the chatgpt.py by specify the question</b>