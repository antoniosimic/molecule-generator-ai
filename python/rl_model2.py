from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, AllChem
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Za MSELoss
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import random

# Akcijski prostor (možete ga proširiti i poboljšati)
actions = [
    'C', 'O', 'N', 'F', 'Cl', 'Br', # Dodao sam Cl i Br kao primjer
    'S', # Sumpor
    '=', # Dvostruka veza (zahtijeva drugačiju logiku u execute_action)
    '#', # Trostruka veza (isto)
    '(', ')', # Za grananje
    '1', '2', '3', # Za prstenove
    'OH', 'NH2', 'COOH', # Funkcionalne grupe
    'remove_atom_idx_0', # Ukloni atom na indeksu 0 (primjer preciznije akcije)
    'add_bond_idx1_idx2_single', # Primjer kompleksnije akcije
    'remove_last'
]
# GORNJI AKCIJSKI PROSTOR JE SAMO PRIMJER POTREBNIH PROMJENA,
# ZAHTIJEVA VELIKU PRERADU `execute_action` FUNKCIJE!
# Za sada ćemo se držati vašeg originalnog jednostavnog akcijskog prostora
# da se fokusiramo na RL algoritam.

original_actions = [ # Vraćamo se na vaš originalni da kod ispod radi
    'C', 'O', 'N', 'F',
    'OH', 'NH2', 'COOH',
    'remove_last'
]


def calculate_reward_with_similarity(original_smiles, modified_smiles, similarity_threshold=0.85, penalty_factor=1.1):
    orig_mol = Chem.MolFromSmiles(original_smiles)
    mod_mol = Chem.MolFromSmiles(modified_smiles)
    if orig_mol is None or mod_mol is None:
        return -1.0 # Vraćamo float
    mod_qed = QED.qed(mod_mol)
    fp_orig = AllChem.GetMorganFingerprintAsBitVect(orig_mol, radius=2, nBits=2048)
    fp_mod  = AllChem.GetMorganFingerprintAsBitVect(mod_mol, radius=2, nBits=2048)
    similarity = DataStructs.TanimotoSimilarity(fp_orig, fp_mod)
    
    reward = mod_qed
    # Kazna za preveliku različitost (možda želite ovo ukloniti ili promijeniti za scaffold hopping)
    if similarity < similarity_threshold:
        penalty = penalty_factor * (similarity_threshold - similarity)
        reward -= penalty
    
    # Dodatna kazna za nevaljane SMILES (iako već gore vraćamo -1)
    if Chem.MolFromSmiles(modified_smiles) is None:
        reward -= 2.0 # Veća kazna
        
    # Nagrada za validnost (mali pozitivni bonus ako je molekula validna)
    if mod_mol is not None:
        reward += 0.1
        
    return float(reward)


def get_state(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=100) # Možda povećati nBits
        arr = np.zeros((1,), dtype=np.int8) # Ispravka za DataStructs.ConvertToNumpyArray
        DataStructs.ConvertToNumpyArray(fp, arr) # arr će biti popunjen
        # arr treba biti prave veličine prije poziva, ili koristite np.array(fp)
        np_fp = np.array(list(fp.ToBitString())).astype(np.float32)
        if len(np_fp) != 100: # Provjera ako je nBits=100
             # Fallback ako ToBitString ne radi kako očekujemo za fiksnu duljinu
             # ili ako je nBits promijenjen
             temp_arr = np.zeros((100,), dtype=np.float32)
             for i in range(min(len(np_fp), 100)): # Osiguravamo da ne premašimo granice
                 temp_arr[i] = np_fp[i]
             return torch.tensor(temp_arr, dtype=torch.float32)
        return torch.tensor(np_fp, dtype=torch.float32)
    else:
        return torch.zeros(100, dtype=torch.float32)

# Actor Mreža (vaša postojeća RLAgent)
class Actor(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x, dim=-1) # Koristimo F.softmax

# Critic Mreža
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1) # Izlaz je jedna vrijednost (V(s))
        )
    def forward(self, x):
        return self.fc(x)

# execute_action (VAŠA FUNKCIJA - PREPORUČUJEM VELIKU PRERADU ZA BOLJE REZULTATE)
# Za sada ostavljamo vašu originalnu da se fokusiramo na RL dio.
# Pripazite: Ova funkcija je kemijski naivna!
def execute_action(smiles, action_str): # Preimenovano u action_str da se ne miješa s globalnom listom
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles 
    editable_mol = Chem.RWMol(mol)
    num_atoms_before_action = editable_mol.GetNumAtoms()

    if action_str == 'remove_last':
        if num_atoms_before_action > 1: # Treba barem jedan atom da ostane ako je to cilj
            editable_mol.RemoveAtom(num_atoms_before_action - 1)
    elif action_str in ['C', 'O', 'N', 'F']:
        new_atom_idx = editable_mol.AddAtom(Chem.Atom(action_str))
        if num_atoms_before_action > 0: # Ako je molekula bila prazna, ne možemo se vezati
            # Nasumično povezivanje je problematično. Pokušajmo se vezati na zadnji postojeći atom ako postoji
            connect_idx = random.randint(0, num_atoms_before_action -1) if num_atoms_before_action > 0 else -1
            if connect_idx != -1 :
                 # Provjeri valencije prije dodavanja veze - ovo je kompleksno i RDKit će pokušati sanirati
                try:
                    editable_mol.AddBond(connect_idx, new_atom_idx, Chem.BondType.SINGLE)
                except RuntimeError: # Može se dogoditi ako RDKit ne može formirati vezu
                    pass # Ignoriraj grešku pri dodavanju veze za sada
    elif action_str in ['OH', 'NH2', 'COOH']: # Jako pojednostavljeno!
        group_atom_symbol = action_str[0] 
        new_atom_idx = editable_mol.AddAtom(Chem.Atom(group_atom_symbol))
        if num_atoms_before_action > 0:
            connect_idx = random.randint(0, num_atoms_before_action -1) if num_atoms_before_action > 0 else -1
            if connect_idx != -1:
                try:
                    editable_mol.AddBond(connect_idx, new_atom_idx, Chem.BondType.SINGLE)
                except RuntimeError:
                    pass
    
    try:
        # Pokušaj sanirati molekulu (npr. postaviti ispravne naboje, vodike)
        Chem.SanitizeMol(editable_mol)
        new_smiles = Chem.MolToSmiles(editable_mol, isomericSmiles=True) # Koristi isomericSmiles
        # Dodatna provjera valjanosti nakon konverzije u SMILES
        if Chem.MolFromSmiles(new_smiles) is None:
            return smiles # Vrati originalni ako novi SMILES nije valjan
        return new_smiles
    except Exception:
        return smiles # Vrati originalni SMILES ako dođe do bilo kakve greške

def train_grpo_model(initial_smiles, num_epochs=500, group_size=5, 
                     actor_lr=1e-4, critic_lr=1e-3, gamma=0.99):
    
    action_space = original_actions # Koristimo originalni jednostavni akcijski prostor
    state_size = 100
    action_size = len(action_space)

    actor = Actor(state_size, action_size)
    critic = Critic(state_size)
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    best_overall_smiles = initial_smiles
    best_overall_qed = QED.qed(Chem.MolFromSmiles(initial_smiles)) if Chem.MolFromSmiles(initial_smiles) else -1.0
    
    all_generated_smiles_qed = [] # Za praćenje svih generiranih

    current_smiles = initial_smiles
    
    for epoch in tqdm(range(num_epochs), desc="Training GRPO-like Model"):
        state_tensor = get_state(current_smiles)
        if torch.all(state_tensor == 0) and current_smiles != "": # Prazan SMILES je OK, ali neispravan nije
            print(f"Epoch {epoch+1}: Invalid current_smiles: '{current_smiles}'. Resetting to 'CCO'.")
            current_smiles = "CCO" # Reset na poznati validan SMILES
            state_tensor = get_state(current_smiles)
            if torch.all(state_tensor == 0): # Ako i CCO ne uspije
                print("CRITICAL: Fallback CCO failed to produce valid state. Exiting.")
                return {"best_smiles": initial_smiles, "best_qed": best_overall_qed, "all_generated": all_generated_smiles_qed}


        log_probs_group = []
        rewards_group = []
        actions_in_group = [] # Za odabir najbolje akcije kasnije
        new_smiles_group = []

        # 1. Grupno uzorkovanje i prikupljanje podataka
        for _ in range(group_size):
            action_probs_dist = actor(state_tensor)
            # Ponekad action_probs_dist može sadržavati NaN ili Inf ako je mreža nestabilna
            if torch.isnan(action_probs_dist).any() or torch.isinf(action_probs_dist).any():
                print(f"Warning: NaN/Inf in action probabilities at epoch {epoch}. Skipping group sample.")
                continue # Preskoči ovaj uzorak iz grupe

            try:
                m = Categorical(action_probs_dist)
                action_idx = m.sample()
                log_prob_action = m.log_prob(action_idx)
            except ValueError as e:
                print(f"Warning: ValueError in Categorical distribution: {e}. Probs: {action_probs_dist}. Skipping group sample.")
                continue


            action_str = action_space[action_idx.item()]
            new_s = execute_action(current_smiles, action_str)
            
            # Koristimo initial_smiles kao referencu za sličnost kroz cijeli trening jedne molekule
            # ili current_smiles da bi se nagrađivala postepena modifikacija?
            # Za "optimizaciju" postojeće molekule, current_smiles je bolja referenca za "lokalnu" sličnost.
            # Ako želimo ostati blizu originala, onda initial_smiles.
            # Trenutna funkcija `calculate_reward_with_similarity` uzima `original_smiles` i `modified_smiles`.
            # Ovdje bi `current_smiles` bio `original_smiles` za taj korak, a `new_s` bi bio `modified_smiles`.
            reward = calculate_reward_with_similarity(current_smiles, new_s)

            log_probs_group.append(log_prob_action)
            rewards_group.append(reward)
            actions_in_group.append(action_idx.item()) # Pamtimo indeks akcije
            new_smiles_group.append(new_s) # Pamtimo novi SMILES

            # Zabilježi sve generirane valjane molekule
            temp_mol = Chem.MolFromSmiles(new_s)
            if temp_mol:
                all_generated_smiles_qed.append((new_s, QED.qed(temp_mol)))


        if not rewards_group: # Ako nijedan uzorak u grupi nije bio uspješan (npr. zbog NaN)
            print(f"Epoch {epoch+1}: No valid samples in group. Skipping update.")
            continue

        # 2. Izračun prednosti (GRPO stil) i ciljeva za Critic
        rewards_tensor = torch.tensor(rewards_group, dtype=torch.float32)
        
        # Critic predviđa vrijednost TRENUTNOG stanja (state_tensor)
        # Ovaj V(s) će se koristiti kao baseline za advantage, ili kao target za učenje Critica
        # Cilj za Critica će biti prosječna nagrada dobivena iz ovog stanja.
        state_value_prediction = critic(state_tensor).squeeze() # Ukloni nepotrebnu dimenziju

        # Prednost: koliko je svaka nagrada bolja/lošija od Criticove procjene (ili grupnog prosjeka)
        # Možemo koristiti V(s) kao baseline ili empirijski prosjek grupe.
        # Korištenje V(s) kao baseline (A_t = r_t - V(s_t)) je češće u A2C.
        # GRPO-like: advantages = rewards_tensor - rewards_tensor.mean() 
        advantages = rewards_tensor - state_value_prediction.detach() # Odvoji da ne utječe na Criticov gradijent kroz Actora


        # 3. Actor Loss
        # Ponderiramo negativni log_prob s prednošću
        # .clone().detach() za advantage da se osigura da se ne mijenja ako je korišten state_value_prediction
        actor_loss = - (torch.stack(log_probs_group) * advantages).mean() 
        
        # 4. Critic Loss
        # Critic uči predviđati prosječnu nagradu koja se dobije iz trenutnog stanja,
        # ili individualne nagrade ako bismo imali parove (stanje, nagrada)
        # Ovdje, neka Critic pokuša predvidjeti prosjek nagrada iz grupe za dano stanje,
        # ili još bolje, neka predviđa vrijednost koja minimizira TD error (što smo koristili za advantage)
        # Dakle, V(s) treba biti što bliže R (ili R + gamma*V(s'))
        # U ovom batch/group pristupu, cilj za V(s) je prosječna nagrada grupe.
        critic_target = rewards_tensor.mean() # Jednostavan target za V(s)
        critic_loss = F.mse_loss(state_value_prediction, critic_target.detach())

        # 5. Ažuriranje mreža
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5) # Opcionalno: Gradijent clipping
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5) # Opcionalno: Gradijent clipping
        critic_optimizer.step()
        
        # 6. Odabir sljedećeg stanja
        # Možemo odabrati molekulu iz grupe koja je dala najveću nagradu
        if rewards_group: # Ako je grupa imala ikakve nagrade
            best_reward_in_group = max(rewards_group)
            best_idx_in_group = rewards_group.index(best_reward_in_group)
            
            # Ovdje je važno kako se definira 'current_reward' koji se uspoređuje.
            # Za sada, samo prelazimo na najbolju molekulu iz grupe ako je bolja od trenutne NAJBOLJE globalno.
            # Ili, možemo samo prijeći na najbolju iz grupe da nastavimo istraživanje.
            # Odlučimo se za: prijeći na najbolju iz grupe da se nastavi istraživanje od tamo.
            current_smiles = new_smiles_group[best_idx_in_group]

            # Ažuriranje globalno najbolje molekule
            current_mol_for_qed = Chem.MolFromSmiles(current_smiles)
            if current_mol_for_qed:
                current_qed_val = QED.qed(current_mol_for_qed)
                if current_qed_val > best_overall_qed:
                    best_overall_qed = current_qed_val
                    best_overall_smiles = current_smiles
                    print(f"\nEpoch {epoch+1}: 🎉 New Best Overall! SMILES: {best_overall_smiles}, QED: {best_overall_qed:.4f}")

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
            print(f"  Current SMILES for next step: {current_smiles}")
            print(f"  Rewards in group: {[f'{r:.2f}' for r in rewards_group]}")


    # Sortiraj sve generirane prema QED i vrati top N ako je potrebno
    all_generated_smiles_qed.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Finished training. Best overall SMILES: {best_overall_smiles}, QED: {best_overall_qed:.4f}")
    return {
        "best_smiles": best_overall_smiles, 
        "best_qed": best_overall_qed,
        "all_generated_top_10": all_generated_smiles_qed[:10] # Prvih 10 kao primjer
    }

if __name__ == '__main__':
    # Primjer pokretanja (pripazite, ovo može dugo trajati)
    initial_aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O" 
    # initial_ethanol = "CCO" # Jednostavnija molekula za brže testiranje
    
    print(f"Starting GRPO-like training with: {initial_aspirin}")
    results = train_grpo_model(initial_aspirin, num_epochs=1000, group_size=8, actor_lr=5e-5, critic_lr=1e-4) # Smanjen broj epoha za test
    
    print("\n--- Training Results ---")
    print(f"Best Generated SMILES: {results['best_smiles']}")
    print(f"Best QED: {results['best_qed']:.4f}")
    
    print("\nTop 10 Generated Molecules (SMILES, QED):")
    if results['all_generated_top_10']:
        for i, (smiles, qed_val) in enumerate(results['all_generated_top_10']):
            print(f"{i+1}. {smiles} (QED: {qed_val:.4f})")
    else:
        print("No valid molecules were generated and recorded in the top list.")