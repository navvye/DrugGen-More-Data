import os
import logging
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Global logging
def setup_logging(output_file):
    log_filename = os.path.splitext(output_file)[0] + '.log'
    # Remove any existing handlers
    logging.getLogger().handlers.clear()

    # File handler for logging to file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Stream handler for logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add handlers to the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # root logger level
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

class BindingAffinityDataset(Dataset):
    def __init__(self, smiles_data, seq_data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        # Prepare training examples
        for smiles, seq in zip(smiles_data, seq_data):
            text = f"<|startoftext|><P>{seq}<L>{smiles}<|endoftext|>"
            self.inputs.append(text)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        text = self.inputs[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Squeeze to remove batch dimension added by tokenizer
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        # Labels are the same as input_ids for language modeling
        item['labels'] = item['input_ids'].clone()
        
        return item

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer from HuggingFace."""
    logging.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise RuntimeError(f"Failed to load model and tokenizer: {e}")

def load_training_data():
    """Load and prepare the binding affinity dataset."""
    logging.info("Loading binding affinity dataset...")
    try:
        ds = load_dataset("jglaser/binding_affinity").sort("neg_log10_affinity_M", reverse=True)
        return ds['train']['smiles_can'][0:10000], ds['train']['seq'][0:10000]
    except Exception as e:
        logging.error(f"Error loading binding affinity dataset: {e}")
        raise RuntimeError(f"Failed to load binding affinity dataset: {e}")

def train_model(model, tokenizer, smiles_data, seq_data, 
               batch_size=8, num_epochs=1, learning_rate=1e-4,
               max_grad_norm=1.0, warmup_steps=0, weight_decay=0.01):
    """Train the model on the binding affinity dataset."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.to(device)
    
    # Create dataset and dataloader
    dataset = BindingAffinityDataset(smiles_data, seq_data, tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * num_epochs
    
    # Initialize learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    best_loss = float('inf')
    training_stats = []
    
    logging.info(f"Starting training on {device}")
    logging.info(f"Total training steps: {total_steps}")
    
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch statistics
        epoch_loss = np.mean(epoch_losses)
        logging.info(f"Epoch {epoch + 1} - Average loss: {epoch_loss:.4f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save model checkpoint
            checkpoint_path = f"model_checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            logging.info(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Store training stats
        training_stats.append({
            'epoch': epoch + 1,
            'average_loss': epoch_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })
    
    # Save final training stats
    stats_df = pd.DataFrame(training_stats)
    stats_df.to_csv('training_stats.csv', index=False)
    logging.info("Saved training statistics to training_stats.csv")
    
    return model

class SMILESGenerator:
    def __init__(self, model, tokenizer, output_file="generated_SMILES.txt"):
        self.output_file = output_file

        # Generation parameters
        self.config = {
            "generation_kwargs": {
                "do_sample": True,
                "top_k": 9,
                "max_length": 1024,
                "top_p": 0.9,
                "num_return_sequences": 10
            },
            "max_retries": 30  # Max retry limit to avoid infinite loops
        }

        # Initialize model and tokenizer
        self.model = model
        self.tokenizer = tokenizer

        self.generation_kwargs = self.config["generation_kwargs"]
        self.generation_kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

    def generate_smiles(self, sequence, num_generated):
        """Generate unique SMILES with a retry limit to avoid infinite loops."""
        generated_smiles_set = set()
        prompt = f"<|startoftext|><P>{sequence}<L>"
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        retries = 0

        logging.info(f"Generating SMILES for sequence: {sequence[:10]}...")

        while len(generated_smiles_set) < num_generated:
            if retries >= self.config["max_retries"]:
                logging.warning("Max retries reached. Returning what has been generated so far.")
                break

            sample_outputs = self.model.generate(encoded_prompt, **self.generation_kwargs)
            for sample_output in sample_outputs:
                output_decode = self.tokenizer.decode(sample_output, skip_special_tokens=False)
                try:
                    generated_smiles = output_decode.split("<L>")[1].split("<|endoftext|>")[0]
                    if generated_smiles not in generated_smiles_set:
                        generated_smiles_set.add(generated_smiles)
                except (IndexError, AttributeError) as e:
                    logging.warning(f"Failed to parse SMILES due to error: {str(e)}. Skipping.")
                    continue

            retries += 1

        logging.info(f"SMILES generation for sequence completed. Generated {len(generated_smiles_set)} SMILES.")
        return list(generated_smiles_set)

    def generate_smiles_data(self, list_of_sequences, num_generated=10):
        """Generate SMILES data for sequences."""
        if not list_of_sequences:
            raise ValueError("`list_of_sequences` must be provided.")

        data = []
        for sequence in list_of_sequences:
            smiles = self.generate_smiles(sequence, num_generated)
            data.append({"sequence": sequence, "SMILES": smiles})

        logging.info(f"Completed SMILES generation for {len(data)} entries.")
        return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate unique SMILES for protein sequences.")
    parser.add_argument("--sequences", nargs="*", help="Protein sequences (space-separated).")
    parser.add_argument("--num_generated", type=int, default=10, help="Number of unique SMILES to generate per input.")
    parser.add_argument("--output_file", type=str, default="generated_SMILES.txt", help="Output file name.")
    parser.add_argument("--train", action="store_true", help="Train the model on binding affinity dataset before generation.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    args = parser.parse_args()

    if not args.output_file.endswith('.txt'):
        args.output_file += '.txt'

    setup_logging(args.output_file)

    if not args.sequences and not args.train:
        raise ValueError("You must provide protein sequences or use --train flag.")

    model_name = "alimotahharynia/DrugGen"
    model, tokenizer = load_model_and_tokenizer(model_name)

    if args.train:
        logging.info("Loading training data...")
        smiles_data, seq_data = load_training_data()
        logging.info(f"Loaded {len(smiles_data)} training examples")
        model = train_model(
            model, 
            tokenizer, 
            smiles_data, 
            seq_data,
            batch_size=args.batch_size
        )

    # Initialize the generator
    generator = SMILESGenerator(model, tokenizer, output_file=args.output_file)
    logging.info("Starting SMILES generation process...")
    
    # Generate SMILES data if sequences provided
    if args.sequences:
        df = generator.generate_smiles_data(
            list_of_sequences=args.sequences,
            num_generated=args.num_generated
        )
        
        # Save the output
        df.to_csv(args.output_file, sep="\t", index=False)
        print(f"Generated SMILES saved to {args.output_file}")
