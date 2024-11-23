# DrugGen: Advancing Drug Discovery with Large Language Models and Reinforcement Learning Feedback
DrugGen is a GPT-2 based model specialized for generating drug-like SMILES structures based on protein sequence. The model leverages the characteristics of approved drug targets and has been trained through both supervised fine-tuning and reinforcement learning techniques to enhance its ability to generate chemically valid, safe, and effective structures.

![Figure1](https://github.com/user-attachments/assets/6fceee2c-950a-41b3-a841-9664a4dd41fd)


## Model Details
-  Model Name: DrugGen
-  Training Paradigm: Supervised Fine-Tuning (SFT) + Proximal Policy Optimization (PPO)
-  Input: Protein Sequence
-  Output: SMILES Structure
-  Training Libraries: Hugging Face’s transformers and Transformer Reinforcement Learning (TRL)
-  Model Sources: liyuesen/druggpt
-  Training data: alimotahharynia/approved_drug_target

## How to Get Started with the Model
Adjust the `num_generated` parameter to specify the number of unique protein SMILES you wish to generate.
```python
from drugGen_generator import SMILESGenerator

if __name__ == "__main__":
    
    # Initialize the generator
    generator = SMILESGenerator()

    # Example input (use either list_of_sequences or list_of_uniprot_ids)
    list_of_sequences = [
        "MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGAQLFAQSYNSSAEQVLFQSVAASWAHDTNITAENARRQEEAALLSQEFAEAWGQKAKELYEPIWQNFTDPQLRRIIGAVRTLGSANLPLAKRQQYNALLSNMSRIYSTAKVCLPNKTATCWSLDPDLTNILASSRSYAMLLFAWEGWHNAAGIPLKPLYEDFTALSNEAYKQDGFTDTGAYWRSWYNSPTFEDDLEHLYQQLEPLYLNLHAFVRRALHRRYGDRYINLRGPIPAHLLGDMWAQSWENIYDMVVPFPDKPNLDVTSTMLQQGWNATHMFRVAEEFFTSLELSPMPPEFWEGSMLEKPADGREVVCHASAWDFYNRKDFRIKQCTRVTMDQLSTVHHEMGHIQYYLQYKDLPVSLRRGANPGFHEAIGDVLALSVSTPEHLHKIGLLDRVTNDTESDINYLLKMALEKIAFLPFGYLVDQWRWGVFSGRTPPSRYNFDWWYLRTKYQGICPPVTRNETHFDAGAKFHVPNVTPYIRYFVSFVLQFQFHEALCKEAGYEGPLHQCDIYRSTKAGAKLRKVLQAGSSRPWQEVLKDMVGLDALDAQPLLKYFQPVTQWLQEQNQQNGEVLGWPEYQWHPPLPDNYPEGIDLVTDEAEASKFVEEYDRTSQVVWNEYAEANWNYNTNITTETSKILLQKNMQIANHTLKYGTQARKFDVNQLQNTTIKRIIKKVQDLERAALPAQELEEYNKILLDMETTYSVATVCHPNGSCLQLEPDLTNVMATSRKYEDLLWAWEGWRDKAGRAILQFYPKYVELINQAARLNGYVDAGDSWRSMYETPSLEQDLERLFQELQPLYLNLHAYVRRALHRHYGAQHINLEGPIPAHLLGNMWAQTWSNIYDLVVPFPSAPSMDTTEAMLKQGWTPRRMFKEADDFFTSLGLLPVPPEFWNKSMLEKPTDGREVVCHASAWDFYNGKDFRIKQCTTVNLEDLVVAHHEMGHIQYFMQYKDLPVALREGANPGFHEAIGDVLALSVSTPKHLHSLNLLSSEGGSDEHDINFLMKMALDKIAFIPFSYLVDQWRWRVFDGSITKENYNQEWWSLRLKYQGLCPPVPRTQGDFDPGAKFHIPSSVPYIRYFVSFIIQFQFHEALCQAAGHTGPLHKCDIYQSKEAGQRLATAMKLGFSRPWPEAMQLITGQPNMSASAMLSYFKPLLDWLRTENELHGEKLGWPQYNWTPNSARSEGPLPDSGRVSFLGLDLDAQQARVGQWLLLFLGIALLVATLGLSQRLFSIRHRSLHRHSHGPQFGSEVELRHS"
    ]
    list_of_uniprot_ids = ["P12821", "P37231"]

    # Generate SMILES data for sequences
    # df = generator.generate_smiles_data(list_of_sequences=list_of_sequences, num_generated=10)

    # Generate SMILES data for UniProt IDs
    df = generator.generate_smiles_data(list_of_uniprot_ids=list_of_uniprot_ids, num_generated=2)

    # Save the output
    output_file = "seq_SMILES.txt"
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Generated SMILES saved to {output_file}")
    print(df)
```
## How to Use Customized Valid Structure Assessor
Here’s how to use `check_smiles.py` to validate a SMILES string:
```python
from check_smiles import check_smiles

# Example SMILES
smiles = "C1=CC=CC=C1"  # Benzene, a valid SMILES
results = check_smiles(smiles)

# Display results
if results:
    print("Issues detected:")
    for penalty, explanation in results:
        print(f"Penalty: {penalty} - {explanation}")
else:
    print("SMILES is valid.")
```
## Citation
If you use this model in your research, please cite our paper:
```
@misc{sheikholeslami2024druggenadvancingdrugdiscovery,
      title={DrugGen: Advancing Drug Discovery with Large Language Models and Reinforcement Learning Feedback}, 
      author={Mahsa Sheikholeslami and Navid Mazrouei and Yousof Gheisari and Afshin Fasihi and Matin Irajpour and Ali Motahharynia},
      year={2024},
      eprint={2411.14157},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2411.14157}, 
}
```
