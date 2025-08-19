# rl_model_selfies_gae.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, AllChem
from tqdm import tqdm
import random
import selfies # Import the selfies library

# --- Configuration ---
STATE_SIZE = 256 
HIDDEN_SIZE = 256 
ACTION_SLEEP_TIME = 0.001 
MAX_SELFIES_LEN = 40 

# --- SELFIES Setup ---
alphabet = list(selfies.get_semantic_robust_alphabet()) 
action_space = alphabet + ['[nop]', '[del]'] 
action_map = {action: i for i, action in enumerate(action_space)}
index_map = {i: action for action, i in action_map.items()}
ACTION_SIZE = len(action_space)

print(f"Using SELFIES alphabet size: {len(alphabet)}")
print(f"Total action space size (alphabet + special): {ACTION_SIZE}")

# --- Helper Functions ---

def smiles_to_selfies(smiles):
    try:
        return selfies.encoder(smiles)
    except Exception as e:
        return None

def selfies_to_smiles(s):
    try:
        return selfies.decoder(s)
    except Exception as e:
        return None

def get_state_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=STATE_SIZE)
            np_fp = np.array(list(fp.ToBitString())).astype(np.float32)
            if len(np_fp) == STATE_SIZE:
                return torch.tensor(np_fp, dtype=torch.float32)
            else:
                padded_fp = np.zeros(STATE_SIZE, dtype=np.float32)
                length_to_copy = min(len(np_fp), STATE_SIZE)
                padded_fp[:length_to_copy] = np_fp[:length_to_copy]
                return torch.tensor(padded_fp, dtype=torch.float32)
        except Exception as e:
            return torch.zeros(STATE_SIZE, dtype=torch.float32)
    else:
        return torch.zeros(STATE_SIZE, dtype=torch.float32)

# REVISED REWARD FUNCTION - UNCONDITIONAL [NOP] PENALTY
def calculate_reward(original_smiles_ref, current_smiles_for_comparison, next_selfies, 
                     is_nop_action, # Novi parametar
                     similarity_target=0.7, 
                     qed_absolute_scale=0.05,       # VRLO MALI scale za apsolutni QED
                     qed_improvement_scale=40.0,    # JOŠ POVEĆAN scale za QED *poboljšanje*
                     valid_selfies_bonus=0.05,
                     similarity_met_bonus=0.0,      
                     hard_penalty_for_invalid=-3.5, # Jača kazna za nevaljane
                     penalty_below_similarity_target=-3.0, # Jača kazna za pad ispod praga sličnosti
                     nop_action_penalty=-0.3): # UNCONDITIONAL Kazna za [nop]

    next_smiles = selfies_to_smiles(next_selfies)

    if next_smiles is None:
        return float(hard_penalty_for_invalid) 
    
    next_mol = Chem.MolFromSmiles(next_smiles)
    if next_mol is None:
        return float(hard_penalty_for_invalid - 0.5)

    try:
        qed_next = QED.qed(next_mol)
    except Exception:
        return float(hard_penalty_for_invalid - 0.2)

    qed_current = 0.0
    if current_smiles_for_comparison:
        current_mol_for_comparison = Chem.MolFromSmiles(current_smiles_for_comparison)
        if current_mol_for_comparison:
            try:
                qed_current = QED.qed(current_mol_for_comparison)
            except Exception:
                pass 

    orig_ref_mol = Chem.MolFromSmiles(original_smiles_ref)
    similarity = 0.0
    if orig_ref_mol:
        try:
            fp_orig_ref = AllChem.GetMorganFingerprintAsBitVect(orig_ref_mol, radius=2, nBits=512)
            fp_next = AllChem.GetMorganFingerprintAsBitVect(next_mol, radius=2, nBits=512)
            similarity = DataStructs.TanimotoSimilarity(fp_orig_ref, fp_next)
        except Exception:
            similarity = 0.0
    else: 
        reward = (qed_next * qed_absolute_scale) + \
                 ((qed_next - qed_current) * qed_improvement_scale) + \
                 valid_selfies_bonus
        if is_nop_action: # Primijeni [nop] kaznu i ovdje ako je primjenjivo
            reward += nop_action_penalty
        return float(reward)

    reward = 0.0
    reward += valid_selfies_bonus

    if similarity < similarity_target:
        reward += penalty_below_similarity_target
    else: 
        reward += similarity_met_bonus 
        reward += (qed_next * qed_absolute_scale) 
        reward += ((qed_next - qed_current) * qed_improvement_scale)

    # Primijeni kaznu za [nop] akciju BEZUVJETNO ako je odabrana
    if is_nop_action:
        reward += nop_action_penalty
        # print(f"Applied unconditional [nop] penalty. Reward adjusted by: {nop_action_penalty}")

    return float(reward)


# REVISED EXECUTE_SELFIES_ACTION (ista kao prethodna verzija)
def execute_selfies_action(current_selfies, action_index):
    action_token = index_map.get(action_index)
    if action_token is None:
        return current_selfies 

    symbols = list(selfies.split_selfies(current_selfies))
    num_symbols = len(symbols)

    if action_token == '[nop]':
        return current_selfies 
    elif action_token == '[del]':
        if num_symbols > 1: 
            del_idx = random.randrange(num_symbols)
            del symbols[del_idx]
    else: 
        rand_choice = random.random()
        append_chance = 0.2 if num_symbols > MAX_SELFIES_LEN * 0.7 else 0.33

        if num_symbols > 0 and rand_choice < 0.4: 
            replace_idx = random.randrange(num_symbols)
            symbols[replace_idx] = action_token
        elif rand_choice < (0.4 + append_chance) and num_symbols < MAX_SELFIES_LEN : 
            symbols.append(action_token)
        elif num_symbols < MAX_SELFIES_LEN : 
            insert_idx = random.randrange(num_symbols + 1)
            symbols.insert(insert_idx, action_token)
            
    new_selfies_str = "".join(symbols)

    if selfies.len_selfies(new_selfies_str) > MAX_SELFIES_LEN:
        return current_selfies 

    return new_selfies_str


# --- Actor-Critic Networks (ostaju iste) ---
class Actor(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=HIDDEN_SIZE):
        super(Actor, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size) 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3_action = nn.Linear(hidden_size // 2, action_size) 

    def forward(self, state):
        x = self.layer_norm(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3_action(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        super(Critic, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size) 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3_value = nn.Linear(hidden_size // 2, 1) 

    def forward(self, state):
        x = self.layer_norm(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3_value(x)
        return value

# --- GAE Calculation (ostaje ista) ---
def calculate_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    num_steps = len(rewards)
    
    for t in reversed(range(num_steps)):
        if dones[t]: 
            delta = rewards[t] - values[t] 
            last_gae_lam = delta 
        else:
            delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            last_gae_lam = delta + gamma * lam * last_gae_lam * (1.0 - dones[t])
        advantages[t] = last_gae_lam
    return advantages

# --- Training Function ---
def train_selfies_gae_model(initial_smiles,
                         num_updates=500,        
                         trajectory_len=128,     
                         epochs_per_update=10,   
                         batch_size=64,          
                         actor_lr=3e-5,          
                         critic_lr=1e-4,
                         gamma=0.99,             
                         lam=0.95,               
                         entropy_coef=0.12, # POVEĆAN DEFAULT ENTROPY COEFFICIENT
                         similarity_target_for_reward=0.7,
                         qed_absolute_scale_for_reward=0.05, # NOVI DEFAULT
                         qed_improvement_scale_for_reward=40.0, # NOVI DEFAULT
                         similarity_met_bonus_for_reward=0.0, 
                         hard_penalty_for_invalid_for_reward=-3.5, # NOVI DEFAULT
                         penalty_below_similarity_target_for_reward=-3.0, # NOVI DEFAULT
                         nop_action_penalty_for_reward=-0.3 # NOVI DEFAULT (preimenovan radi jasnoće)
                         ):   

    initial_selfies = smiles_to_selfies(initial_smiles)
    if initial_selfies is None:
        print(f"Error: Could not convert initial SMILES '{initial_smiles}' to SELFIES. Aborting.")
        return {"best_smiles": initial_smiles, "best_qed": -1, "log": "Invalid initial SMILES"}

    actor = Actor(STATE_SIZE, ACTION_SIZE)
    critic = Critic(STATE_SIZE)
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    best_overall_smiles = initial_smiles
    initial_mol_for_qed = Chem.MolFromSmiles(initial_smiles)
    best_overall_qed = QED.qed(initial_mol_for_qed) if initial_mol_for_qed else -1.0
    
    training_log = [] 
    all_generated_smiles_qed = []
    current_selfies = initial_selfies

    print(f"Starting training from: {initial_smiles} (SELFIES: {current_selfies})")
    print(f"Targeting {num_updates} updates with trajectory length {trajectory_len}.")
    print(f"Reward params: sim_target={similarity_target_for_reward}, qed_abs_scale={qed_absolute_scale_for_reward}, qed_imp_scale={qed_improvement_scale_for_reward}, unconditional_nop_penalty={nop_action_penalty_for_reward}")
    print(f"Entropy coefficient: {entropy_coef}")


    for update in tqdm(range(num_updates), desc="Policy Updates"):
        trajectory_states, trajectory_actions, trajectory_log_probs = [], [], []
        trajectory_rewards, trajectory_values, trajectory_dones, trajectory_next_values = [], [], [], []
        
        actor.eval() 
        critic.eval() 
        temp_generated_in_traj = []
        selfies_at_step_start = current_selfies 

        for step in range(trajectory_len):
            current_smiles_for_state_and_reward = selfies_to_smiles(selfies_at_step_start)
            if current_smiles_for_state_and_reward is None:
                selfies_at_step_start = smiles_to_selfies("C") 
                if selfies_at_step_start is None: selfies_at_step_start = '[C]' 
                current_smiles_for_state_and_reward = selfies_to_smiles(selfies_at_step_start)
                state_tensor = get_state_from_smiles(current_smiles_for_state_and_reward) if current_smiles_for_state_and_reward else torch.zeros(STATE_SIZE, dtype=torch.float32)
            else:
                 state_tensor = get_state_from_smiles(current_smiles_for_state_and_reward)

            trajectory_states.append(state_tensor)

            with torch.no_grad():
                action_probs = actor(state_tensor)
                value_estimate = critic(state_tensor)

            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                action_probs = torch.ones_like(action_probs) / ACTION_SIZE 
            
            try:
                dist = Categorical(probs=action_probs) 
                action = dist.sample()
                log_prob = dist.log_prob(action)
            except ValueError as e:
                 action = torch.tensor(action_map['[nop]']) 
                 log_prob = torch.tensor(-np.log(ACTION_SIZE)) 

            action_idx = action.item()
            action_token_selected = index_map.get(action_idx, '[nop]') 
            is_nop = (action_token_selected == '[nop]')

            next_selfies_after_action = execute_selfies_action(selfies_at_step_start, action_idx)
            
            reward = calculate_reward(
                original_smiles_ref=initial_smiles, 
                current_smiles_for_comparison=current_smiles_for_state_and_reward,
                next_selfies=next_selfies_after_action, 
                is_nop_action=is_nop, 
                similarity_target=similarity_target_for_reward,
                qed_absolute_scale=qed_absolute_scale_for_reward, 
                qed_improvement_scale=qed_improvement_scale_for_reward,
                similarity_met_bonus=similarity_met_bonus_for_reward, 
                hard_penalty_for_invalid=hard_penalty_for_invalid_for_reward,
                penalty_below_similarity_target=penalty_below_similarity_target_for_reward,
                nop_action_penalty=nop_action_penalty_for_reward # Proslijeđujemo novu bezuvjetnu kaznu
                # Uklonjeni qed_threshold_for_nop_penalty jer je kazna sada bezuvjetna
            )

            next_smiles_for_state_val = selfies_to_smiles(next_selfies_after_action)
            next_state_tensor_val = get_state_from_smiles(next_smiles_for_state_val) if next_smiles_for_state_val else torch.zeros(STATE_SIZE, dtype=torch.float32)

            with torch.no_grad():
                next_value_estimate = critic(next_state_tensor_val)

            trajectory_actions.append(action)
            trajectory_log_probs.append(log_prob)
            trajectory_rewards.append(torch.tensor(reward, dtype=torch.float32))
            trajectory_values.append(value_estimate.squeeze()) 
            trajectory_next_values.append(next_value_estimate.squeeze()) 
            done_flag = (next_smiles_for_state_val is None) 
            trajectory_dones.append(torch.tensor(done_flag, dtype=torch.float32))

            if next_smiles_for_state_val:
                 temp_mol = Chem.MolFromSmiles(next_smiles_for_state_val)
                 if temp_mol:
                     qed_val = QED.qed(temp_mol)
                     temp_generated_in_traj.append((next_smiles_for_state_val, qed_val))
                     if qed_val > best_overall_qed:
                         fp_orig_ref_check = AllChem.GetMorganFingerprintAsBitVect(initial_mol_for_qed, radius=2, nBits=512)
                         fp_next_check = AllChem.GetMorganFingerprintAsBitVect(temp_mol, radius=2, nBits=512)
                         similarity_check = DataStructs.TanimotoSimilarity(fp_orig_ref_check, fp_next_check)
                         if similarity_check >= similarity_target_for_reward:
                            best_overall_qed = qed_val
                            best_overall_smiles = next_smiles_for_state_val
                            print(f"\nUpdate {update+1}, Step {step+1}: 🎉 New Best! SMILES: {best_overall_smiles}, QED: {best_overall_qed:.4f}, Sim: {similarity_check:.3f}")
            
            if not done_flag: 
                selfies_at_step_start = next_selfies_after_action
            else: 
                selfies_at_step_start = initial_selfies 
        
        current_selfies = selfies_at_step_start
        all_generated_smiles_qed.extend(temp_generated_in_traj)

        if not trajectory_rewards or len(trajectory_rewards) == 0: 
            print(f"Update {update+1}: Trajectory empty or no rewards, skipping GAE and updates.")
            continue
        
        min_len = len(trajectory_rewards) 
        
        rewards_t = torch.stack(trajectory_rewards)[:min_len]
        values_t = torch.stack(trajectory_values)[:min_len]
        next_values_t = torch.stack(trajectory_next_values)[:min_len]
        dones_t = torch.stack(trajectory_dones)[:min_len]
        log_probs_t = torch.stack(trajectory_log_probs)[:min_len]
        states_t = torch.stack(trajectory_states)[:min_len]
        actions_t = torch.stack(trajectory_actions)[:min_len]

        if len(rewards_t) == 0: 
            print(f"Update {update+1}: Trajectory effectively empty. Skipping GAE and updates.")
            continue

        advantages_t = calculate_gae(rewards_t, values_t, next_values_t, dones_t, gamma, lam)
        returns_t = advantages_t + values_t 
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        actor.train() 
        critic.train()
        total_actor_loss, total_critic_loss, total_entropy_loss = 0,0,0
        
        current_trajectory_actual_len = len(rewards_t)
        indices = np.arange(current_trajectory_actual_len) 

        for _ in range(epochs_per_update):
            np.random.shuffle(indices)
            for start in range(0, current_trajectory_actual_len, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                if len(batch_indices) == 0: continue

                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]      

                values_pred = critic(batch_states).squeeze()
                if values_pred.ndim == 0 and len(batch_indices) == 1: values_pred = values_pred.unsqueeze(0)
                elif values_pred.ndim > 0 and len(values_pred) != len(batch_indices) : 
                    print(f"Warning: Mismatch in critic prediction and batch size. values_pred shape: {values_pred.shape}, batch_returns shape: {batch_returns.shape}")
                    continue 
                
                critic_loss = F.mse_loss(values_pred, batch_returns.detach()) 

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5) 
                critic_optimizer.step()

                action_probs_new = actor(batch_states)
                dist_new = Categorical(probs=action_probs_new + 1e-8) 
                batch_log_probs_new = dist_new.log_prob(batch_actions)
                
                actor_loss = - (batch_log_probs_new * batch_advantages.detach()).mean()
                entropy = dist_new.entropy().mean()
                entropy_loss = - entropy_coef * entropy 
                total_actor_loss_batch = actor_loss + entropy_loss

                actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5) 
                actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.item()
        
        num_minibatches = (current_trajectory_actual_len + batch_size -1) // batch_size 
        avg_actor_loss = total_actor_loss / (num_minibatches * epochs_per_update) if num_minibatches > 0 else 0
        avg_critic_loss = total_critic_loss / (num_minibatches * epochs_per_update) if num_minibatches > 0 else 0
        avg_entropy = total_entropy_loss / (num_minibatches * epochs_per_update) if num_minibatches > 0 else 0
        avg_reward_in_traj = rewards_t.mean().item() if current_trajectory_actual_len > 0 else -999

        log_entry = f"Update {update+1}/{num_updates}: Avg Reward: {avg_reward_in_traj:.3f}, Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}, Entropy: {avg_entropy:.3f}"
        print(log_entry)
        training_log.append(log_entry)

    all_generated_smiles_qed.sort(key=lambda x: x[1], reverse=True)

    print(f"\nFinished training. Best overall SMILES found: {best_overall_smiles}, QED: {best_overall_qed:.4f}")
    return {
        "best_smiles": best_overall_smiles,
        "best_qed": best_overall_qed,
        "log": "\n".join(training_log), 
        "all_generated_top_10": all_generated_smiles_qed[:10]
    }

if __name__ == '__main__':
    initial_mol_smiles = "Cc1ccc(cc1)c2cc(c(cc2O)C)O" 
    print(f"Starting SELFIES GAE training with initial SMILES: {initial_mol_smiles}")

    results = train_selfies_gae_model(
        initial_smiles=initial_mol_smiles,
        num_updates=200,       
        trajectory_len=64,      
        epochs_per_update=5,      
        batch_size=32,
        actor_lr=5e-5, 
        critic_lr=1e-4,
        entropy_coef=0.12, # POVEĆAN ENTROPY
        similarity_target_for_reward=0.7, 
        qed_absolute_scale_for_reward=0.05,  # VRLO MALI UTJECAJ APSOLUTNOG QED-a
        qed_improvement_scale_for_reward=40.0, # VRLO JAKO NAGRAĐIVANJE QED POBOLJŠANJA     
        similarity_met_bonus_for_reward=0.0,
        hard_penalty_for_invalid_for_reward=-3.5, 
        penalty_below_similarity_target_for_reward=-3.0,
        nop_action_penalty_for_reward=-0.3 # Bezuvjetna kazna za [nop]
    )

    print("\n--- Training Results ---")
    print(f"Best Generated SMILES: {results['best_smiles']}")
    print(f"Best QED: {results['best_qed']:.4f}")

    print("\nTop 10 Generated Molecules (SMILES, QED):")
    if results.get('all_generated_top_10'):
        for i, (smiles, qed_val) in enumerate(results['all_generated_top_10']):
            print(f"{i+1}. {smiles} (QED: {qed_val:.4f})")
    else:
        print("No valid molecules were generated and recorded.")

