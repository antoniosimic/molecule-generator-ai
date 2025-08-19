from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, AllChem
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

# Extended action space
actions = [
    'C', 'O', 'N', 'F',       # Adding simple atoms
    'OH', 'NH2', 'COOH',       # Adding small functional groups
    'remove_last'            # Removing the last atom
]

# Reward function combining QED and penalizing when similarity is below a threshold
def calculate_reward_with_similarity(original_smiles, modified_smiles, similarity_threshold=0.85, penalty_factor=1.1):
    orig_mol = Chem.MolFromSmiles(original_smiles)
    mod_mol = Chem.MolFromSmiles(modified_smiles)
    if orig_mol is None or mod_mol is None:
        return -1  # Penalize invalid molecules
    mod_qed = QED.qed(mod_mol)
    # Use 2048-bit Morgan fingerprints for a precise similarity measure
    fp_orig = AllChem.GetMorganFingerprintAsBitVect(orig_mol, radius=2, nBits=2048)
    fp_mod  = AllChem.GetMorganFingerprintAsBitVect(mod_mol, radius=2, nBits=2048)
    similarity = DataStructs.TanimotoSimilarity(fp_orig, fp_mod)
    penalty = 0
    if similarity < similarity_threshold:
        penalty = penalty_factor * (similarity_threshold - similarity)
    reward = mod_qed - penalty
    return reward

# Convert SMILES to a state vector using a 100-bit Morgan fingerprint
def get_state(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=100)
        arr = np.zeros((100,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return torch.tensor(arr, dtype=torch.float32)
    else:
        return torch.zeros(100)

# RL Agent: A simple neural network that proposes an action
class RLAgent(nn.Module):
    def __init__(self, input_size, action_size):
        super(RLAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(x, dim=-1)

# Execute the chosen action on the molecule represented by SMILES
def execute_action(smiles, action):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # Return original if molecule is invalid
    editable_mol = Chem.RWMol(mol)
    if action == 'remove_last':
        if editable_mol.GetNumAtoms() > 1:
            editable_mol.RemoveAtom(editable_mol.GetNumAtoms() - 1)
        new_mol = editable_mol.GetMol()
    elif action in ['C', 'O', 'N', 'F']:
        new_atom_idx = editable_mol.AddAtom(Chem.Atom(action))
        num_atoms = editable_mol.GetNumAtoms()
        if num_atoms > 1:
            connect_idx = random.randint(0, num_atoms - 2)
            editable_mol.AddBond(connect_idx, new_atom_idx, Chem.BondType.SINGLE)
        new_mol = editable_mol.GetMol()
    elif action in ['OH', 'NH2', 'COOH']:
        # For simplicity, add only the first character of the functional group
        group_atom = action[0]
        new_atom_idx = editable_mol.AddAtom(Chem.Atom(group_atom))
        num_atoms = editable_mol.GetNumAtoms()
        if num_atoms > 1:
            connect_idx = random.randint(0, num_atoms - 2)
            editable_mol.AddBond(connect_idx, new_atom_idx, Chem.BondType.SINGLE)
        new_mol = editable_mol.GetMol()
    else:
        new_mol = mol
    try:
        new_smiles = Chem.MolToSmiles(new_mol)
        return new_smiles
    except Exception as e:
        return smiles

# Train the RL model starting from the given SMILES.
# Returns a dictionary with the best generated molecule (SMILES) and its QED value.
def train_rl_model(initial_smiles, num_epochs=5000, epsilon=0.0):
    model = RLAgent(input_size=100, action_size=len(actions))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    generated_molecules = []
    current_smiles = initial_smiles
    current_reward = calculate_reward_with_similarity(initial_smiles, initial_smiles)
    
    for epoch in tqdm(range(num_epochs), desc="Training RL Model"):
        state = get_state(current_smiles)
        
        # ε-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, len(actions) - 1)
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            action_probs = model(state)
            distribution = torch.distributions.Categorical(action_probs)
            action_idx = distribution.sample().item()
            loss = -distribution.log_prob(torch.tensor(action_idx, dtype=torch.long))
        
        action = actions[action_idx]
        new_smiles = execute_action(current_smiles, action)
        reward = calculate_reward_with_similarity(current_smiles, new_smiles, similarity_threshold=0.85, penalty_factor=1.1)
        
        # Save valid molecules and their QED values
        mol_temp = Chem.MolFromSmiles(new_smiles)
        if mol_temp:
            qed_val = QED.qed(mol_temp)
            generated_molecules.append((new_smiles, qed_val))
        
        # Update current molecule if reward improves
        if reward > current_reward:
            current_smiles = new_smiles
            current_reward = reward
            print(f"\n✅ Molecule updated: {current_smiles}")
        
        if loss.requires_grad:
            loss = loss * torch.tensor(reward, dtype=torch.float32)
        
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Action: {action}, Reward: {reward:.4f}, SMILES: {new_smiles}")
    
    if generated_molecules:
        best = max(generated_molecules, key=lambda x: x[1])
        return {
            "best_smiles": best[0],
            "best_qed": best[1]
        }
    else:
        return {
            "best_smiles": initial_smiles,
            "best_qed": QED.qed(Chem.MolFromSmiles(initial_smiles)) if Chem.MolFromSmiles(initial_smiles) else 0
        }

