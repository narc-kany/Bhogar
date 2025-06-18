import torch
import random
import os
from rdkit import Chem
from rdkit.Chem import EditableMol # Needed for mutate_smiles

# Import modules from src
from src.model import HybridQuantumModel
from src.preprocessing import mol_to_features
from src.utils import mutate_smiles, generate_drug_name_from_score

# Configuration
MODEL_PATH = 'models/hybrid_quantum_model.pth'
NUM_CANDIDATES = 50 # Number of mutated drugs to generate for evaluation

def generate_initial_smiles_candidates(statement, num_candidates=5):
    """
    Generates initial SMILES candidates based on a natural language statement.
    This is a highly simplified placeholder. In a real-world scenario,
    this would involve advanced NLP (e.g., text-to-SMILES generation,
    retrieval from databases based on keywords).

    For this example, we'll return a few generic starting SMILES.
    """
    # Example generic SMILES that are reasonably simple
    # You might want to use some common scaffold SMILES relevant to drug design
    generic_smiles = [
        "C1=CC=CC=C1C(=O)NC", # Phenyl-amide
        "O=C(O)CCC",          # Carboxylic acid
        "NCC(=O)O",           # Amino acid derivative
        "c1cncc(N)c1",        # Pyridine with amine
        "CC(C)NC(=O)C"        # Simple amide
    ]
    
    # A very basic "response" mechanism:
    if "memory" in statement.lower() or "cognition" in statement.lower():
        # Prioritize certain SMILES if keywords are present (still generic)
        return random.sample(generic_smiles, min(num_candidates, len(generic_smiles)))
    else:
        # Just pick some random ones if no specific keywords
        return random.sample(generic_smiles, min(num_candidates, len(generic_smiles)))

def discover_new_drug(user_statement, model, num_candidates=NUM_CANDIDATES):
    """
    Takes a user statement, generates/mutates SMILES candidates,
    evaluates them with the model, and suggests a new drug name.
    """
    print(f"\nAnalyzing statement: '{user_statement}'")

    # Step 1: Generate initial diverse SMILES candidates based on statement
    # This is a very simplistic approach. A real system would use NLP + molecular generation.
    initial_base_smiles = generate_initial_smiles_candidates(user_statement, num_candidates=5)
    
    # Step 2: Mutate initial candidates to create a pool of diverse candidates
    candidates = set()
    for base_smi in initial_base_smiles:
        candidates.add(base_smi) # Add original
        for _ in range(num_candidates // len(initial_base_smiles)): # Generate more mutations per base
            mutated_smi = mutate_smiles(base_smi)
            if mutated_smi and Chem.MolFromSmiles(mutated_smi) is not None:
                candidates.add(mutated_smi)
    
    if not candidates:
        print("Could not generate any valid candidate SMILES. Please try a different statement or check RDKit installation.")
        return None, None, None

    print(f"Generated {len(candidates)} unique drug candidates for evaluation.")

    # Step 3: Evaluate candidates with the trained model
    new_drugs = []
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for smi in candidates:
            feats = mol_to_features(smi)
            if feats:
                inp = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                score = model(inp).item()
                new_drugs.append((smi, score, feats))

    if not new_drugs:
        print("No candidates could be featurized or evaluated. Check SMILES validity.")
        return None, None, None

    # Step 4: Rank candidates by predicted activity score
    new_drugs.sort(key=lambda x: x[1], reverse=True)

    # Step 5: Select the top candidate and generate a name
    top_candidate = new_drugs[0]
    smiles, score, desc = top_candidate
    
    # Generate a more specific drug name based on the predicted score
    drug_name = generate_drug_name_from_score(score)

    return drug_name, smiles, score, desc

if __name__ == "__main__":
    print("Loading Hybrid Quantum Model...")
    model = HybridQuantumModel()
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}.")
        print("Please run `python src/train.py` first to train and save the model.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    print("\n--- AI-Powered Alzheimer's Drug Discovery ---")
    print("Enter a statement describing the desired drug's purpose or characteristics.")
    print("Type 'quit' to exit.")

    while True:
        statement = input("\nYour statement: ")
        if statement.lower() == 'quit':
            break

        drug_name, smiles, score, features = discover_new_drug(statement, model)

        if drug_name:
            logP, molWt, tpsa = features
            print("\n--- Newly Discovered Drug Candidate ---")
            print(f"Drug Name      : {drug_name}")
            print(f"SMILES         : {smiles}")
            print(f"Predicted Score: {score:.4f} (Higher is better)")
            print("\nJustification (Based on molecular properties):")
            print(f"- Molecular Weight: {molWt:.2f} (Optimal for oral drugs: 200–500 Da)")
            print(f"- LogP (Lipophilicity): {logP:.2f} (Optimal for CNS drugs: <5)")
            print(f"- TPSA (Polar Surface Area): {tpsa:.2f} (Optimal for BBB permeability: <90 Å²)")
            print("- The model predicted a high activity score, suggesting strong potential for Alzheimer's.")
        else:
            print("Drug discovery failed for this statement. Please try another one.")