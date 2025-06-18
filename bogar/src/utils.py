from rdkit import Chem
from rdkit.Chem import EditableMol
import random

def mutate_smiles(smiles):
    """
    Mutates a given SMILES string by adding a carbon atom to a random valid bond.
    Highly simplified for demonstration; real mutation requires sophisticated algorithms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    em = EditableMol(mol)

    # Choose only carbon or nitrogen atoms for bonding
    valid_idxs = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() in ["C", "N"]]
    if not valid_idxs:
        return None

    try:
        idx = random.choice(valid_idxs)
        # Add a new carbon atom and a single bond to the chosen atom
        new_atom_idx = em.AddAtom(Chem.Atom("C"))
        em.AddBond(idx, new_atom_idx, Chem.rdchem.BondType.SINGLE)
        
        mutated_mol = em.GetMol()
        Chem.SanitizeMol(mutated_mol) # Ensure validity
        return Chem.MolToSmiles(mutated_mol, canonical=True)
    except Exception as e:
        # print(f"Mutation failed for SMILES {smiles}: {e}") # Uncomment for debugging
        return None

def generate_drug_name_from_score(score):
    """
    Generates a creative drug name based on the predicted score.
    This is a placeholder and can be made much more sophisticated.
    """
    prefixes = ["Neuro", "Cogni", "Alzene", "Memo", "Synapto", "Dendro"]
    suffixes = ["vex", "zol", "cept", "tone", "mab", "stat", "lyte"]
    
    # Use score to influence the choice
    # For example, higher score could lean towards more "active"-sounding names
    prefix_idx = int(score * (len(prefixes) - 1)) # Scale score to prefix index
    suffix_idx = int((1 - score) * (len(suffixes) - 1)) # Invert for suffix if desired
    
    base_name = prefixes[prefix_idx]
    drug_suffix = suffixes[suffix_idx]
    
    # Add a random number or a short hash for uniqueness
    random_part = random.randint(10, 99)
    
    return f"{base_name}{drug_suffix}-{random_part}"