import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset

class SMILESGenerator:
    def __init__(self):
        
        # Configuration parameters
        self.config = {
            "model_name": "alimotahharynia/DrugGen",
            "dataset_name": "alimotahharynia/approved_drug_target",
            "dataset_key": "uniprot_sequence",
            "generation_kwargs": {
                "do_sample": True,
                "top_k": 9,
                "max_length": 1024,
                "top_p": 0.9,
                "num_return_sequences": 10
            },
            "max_retries": 30  # Max retry limit to avoid infinite loops
        }

        # Load model and tokenizer
        self.model_name = self.config["model_name"]
        self.model, self.tokenizer = self.load_model_and_tokenizer(self.model_name)

        # Load UniProt mapping dataset
        dataset_name = self.config["dataset_name"]
        dataset_key = self.config["dataset_key"]
        self.uniprot_to_sequence = self.load_uniprot_mapping(dataset_name, dataset_key)

        # Adjust generation parameters with token IDs
        self.generation_kwargs = self.config["generation_kwargs"]
        self.generation_kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

    def load_model_and_tokenizer(self, model_name):

        print(f"Loading model and tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        return model, tokenizer

    def load_uniprot_mapping(self, dataset_name, dataset_key):
 
        print(f"Loading dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, dataset_key)
            return {row["UniProt_id"]: row["Sequence"] for row in dataset["uniprot_seq"]}
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

    def generate_smiles(self, sequence, num_generated):
        """
        Generate unique SMILES with a retry limit to avoid infinite loops.
        """
        generated_smiles_set = set()
        prompt = f"<|startoftext|><P>{sequence}<L>"
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        retries = 0

        while len(generated_smiles_set) < num_generated:
            if retries >= self.config["max_retries"]:
                print("Max retries reached. Returning what has been generated so far.")
                break
            
            sample_outputs = self.model.generate(encoded_prompt, **self.generation_kwargs)
            for sample_output in sample_outputs:
                output_decode = self.tokenizer.decode(sample_output, skip_special_tokens=False)
                try:
                    generated_smiles = output_decode.split("<L>")[1].split("<|endoftext|>")[0]
                    if generated_smiles not in generated_smiles_set:
                        generated_smiles_set.add(generated_smiles)
                except IndexError:
                    continue
            
            retries += 1

        return list(generated_smiles_set)

    def generate_smiles_data(self, list_of_sequences=None, list_of_uniprot_ids=None, num_generated=10):
        """
        Generate SMILES data for sequences or UniProt IDs.
        """
        if not list_of_sequences and not list_of_uniprot_ids:
            raise ValueError("Either `list_of_sequences` or `list_of_uniprot_ids` must be provided.")

        # Prepare sequences input
        if list_of_sequences:
            sequences_input = list_of_sequences
        else:
            sequences_input = [
                self.uniprot_to_sequence[uid]
                for uid in list_of_uniprot_ids
                if uid in self.uniprot_to_sequence
            ]

        data = []
        for sequence in sequences_input:
            smiles = self.generate_smiles(sequence, num_generated)
            uniprot_id = next((uid for uid, seq in self.uniprot_to_sequence.items() if seq == sequence), None)
            data.append({"UniProt_id": uniprot_id, "sequence": sequence, "SMILES": smiles})

        return pd.DataFrame(data)