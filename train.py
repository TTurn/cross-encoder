from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime

def read_chatbot_csv(chatbot_csv):
    samples = []
    with open(chatbot_csv, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line_s = line.split("\t")
            if len(line_s) != 3:
                continue
            query, key, value = line_s
            if value not in ['0', '1']:
                continue
            value = int(value)
            samples.append(InputExample(texts=[query, key], label=value))
            samples.append(InputExample(texts=[key, query], label=value))
    return samples

path = ""
train_csv = path + 'train.tsv'
dev_csv = path + 'dev.tsv'
test_csv = path + 'test.tsv'
train_samples = []
dev_samples = []
test_samples = []
train_samples = read_chatbot_csv(train_csv)
dev_samples = read_chatbot_csv(dev_csv)
test_samples = read_chatbot_csv(test_csv)
print(len(train_samples))
print(len(dev_samples))
print(len(test_samples))
print(dev_samples[0].texts,dev_samples[0].label)
print(train_samples[0].texts,train_samples[0].label)

train_batch_size = 32
num_epochs = 1
model_save_path = 'crossencoder_roberta-wwm-ext'
#model_save_path = 'sbert/chinese-roberta-wwm-ext-large'
#model_save_path = 'sbert/chinese-electra-180g-large-discriminator'
#Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
model = CrossEncoder("hfl/chinese-roberta-wwm-ext", num_labels=1, max_length=64,device='cuda')
#model = CrossEncoder("hfl/chinese-electra-180g-large-discriminator", num_labels=1, max_length=64,device='cuda')

#We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

#During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CEBinaryAccuracyEvaluator.from_input_examples(dev_samples)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


