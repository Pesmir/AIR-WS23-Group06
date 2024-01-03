import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# Variables
MAX_LEN = 128
BATCH_SIZE = 8
layers_to_freez = 2
file_path = '../ReviewPreprocessing.ipc'
model_name = 'bert_fine-tuned_1%_model'
data_percent = 100
EPOCHS = 3


data = pd.read_feather(file_path)

tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

data['adjusted_targets'] = data['review/score'] - 1

df_train = data.sample(frac=0.8, random_state=25)
df_val = data.drop(df_train.index)

train_data = ReviewDataset(
    reviews=df_train['preprocessed_review/text'].to_numpy(),
    targets=df_train['adjusted_targets'].to_numpy(),  # Use the adjusted targets
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

val_data = ReviewDataset(
    reviews=df_val['preprocessed_review/text'].to_numpy(),
    targets=df_val['adjusted_targets'].to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_size = int((data_percent/100) * len(train_data))
subset, _ = random_split(train_data, [train_size, len(train_data) - train_size])

train_data_loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_size = int((data_percent/100) * len(val_data))
subset_val, _ = random_split(val_data, [train_size, len(val_data) - train_size])

val_data_loader = DataLoader(
    subset_val,
    batch_size=BATCH_SIZE
)

for name, param in model.named_parameters():
    if name.startswith('bert.encoder.layer'):
        layer_num = int(name.split('.')[3])
        if layer_num < 12 - layers_to_freez:
            param.requires_grad = False

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

TOTAL_STEPS = len(train_data_loader) * EPOCHS

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    train_progress_bar = tqdm(train_data_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Training]', leave=False)

    for batch in train_progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        model.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        train_progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

    avg_train_loss = total_loss / len(train_data_loader)

    model.eval()
    total_eval_loss = 0

    val_progress_bar = tqdm(val_data_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Validation]', leave=False)

    for batch in val_progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )

        loss = outputs.loss
        total_eval_loss += loss.item()

        val_progress_bar.set_postfix({'validation_loss': '{:.3f}'.format(loss.item())})

    avg_val_loss = total_eval_loss / len(val_data_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss} | Validation Loss: {avg_val_loss}')

model.save_pretrained(f'results/{model_name}')