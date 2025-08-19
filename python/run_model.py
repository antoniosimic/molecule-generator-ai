from flask import Flask, request, jsonify
from flask_cors import CORS
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, AllChem, QED
from io import BytesIO
import base64
import traceback # Važno za ispisivanje cijelog traga greške

# Importiramo novu funkciju za model1
from rl_model1 import train_selfies_gae_model as train_rl_model1 # Sada model1 koristi SELFIES GAE
# Pretpostavljamo da rl_model2 (GRPO-like) i ostali (3, 4) imaju svoje definicije
from rl_model2 import train_grpo_model as train_rl_model2 
from rl_model3 import train_rl_model as train_rl_model3 # Pretpostavka: starija implementacija
from rl_model4 import train_rl_model as train_rl_model4 # Pretpostavka: starija implementacija

app = Flask(__name__)
CORS(app)

MODEL_DISPATCHER = {
    "model1": train_rl_model1, # Koristi novu SELFIES GAE funkciju
    "model2": train_rl_model2, # Koristi GRPO-like funkciju
    "model3": train_rl_model3,
    "model4": train_rl_model4,
}

@app.route('/generate', methods=['POST'])
def generate_molecule():
    try:
        data = request.get_json()
        smiles = data.get('smiles')
        model_id = data.get('model_id')

        if not smiles:
            return jsonify({"error": "SMILES string is required"}), 400
        
        if model_id not in MODEL_DISPATCHER:
            return jsonify({
                "error": f"Invalid model_id. Choose one of {list(MODEL_DISPATCHER.keys())}"
            }), 400

        initial_mol = Chem.MolFromSmiles(smiles)
        input_qed = 0.0
        input_img_str = ""
        input_mol_block_str = ""

        if initial_mol:
            input_qed = QED.qed(initial_mol)
            buf = BytesIO()
            img = Draw.MolToImage(initial_mol, size=(300, 300))
            img.save(buf, format="PNG")
            input_img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            try:
                AllChem.EmbedMolecule(initial_mol, randomSeed=0xf00d)
                AllChem.UFFOptimizeMolecule(initial_mol)
                input_mol_block_str = Chem.MolToMolBlock(initial_mol)
            except Exception as e_3d_input:
                print(f"Error generating 3D for input molecule: {e_3d_input}")
                input_mol_block_str = ""
        else:
            print(f"Warning: Could not parse input SMILES: {smiles}")
            # Možda vratiti grešku ako ulazni SMILES nije validan
            # return jsonify({"error": "Invalid input SMILES string"}), 400

        training_function = MODEL_DISPATCHER[model_id]
        result = {} # Inicijalizacija rezultata

        # ISPRAVLJENA LOGIKA POZIVA MODELA
        if model_id == "model1":
            result = training_function(
                initial_smiles=smiles, 
                num_updates=200, 
                trajectory_len=64,
                epochs_per_update=5,
                batch_size=32,
                actor_lr=5e-5,
                critic_lr=1e-4,
                entropy_coef=0.12, # Povećan koeficijent entropije
                similarity_target_for_reward=0.7,
                qed_absolute_scale_for_reward=0.05,
                qed_improvement_scale_for_reward=40.0, 
                similarity_met_bonus_for_reward=0.0,
                hard_penalty_for_invalid_for_reward=-3.5,
                penalty_below_similarity_target_for_reward=-3.0,
                nop_action_penalty_for_reward=-0.3 # Bezuvjetna kazna za [nop]
            )
        # ... (ostali elif blokovi)
        elif model_id == "model2":
            # Parametri za train_grpo_model (kako ste imali prije)
            result = training_function(
                initial_smiles=smiles, 
                num_epochs=1000, # Ili num_updates ako ste i GRPO prebacili na taj stil
                group_size=8      # Primjer
                # Dodajte ostale parametre koje train_grpo_model očekuje
            )
        elif model_id in ["model3", "model4"]:
            # Parametri za stare modele (ako koriste num_epochs, epsilon)
            result = training_function(
                initial_smiles=smiles, 
                num_epochs=5000, 
                epsilon=0.0
            )
        else:
            # Fallback ili greška ako model_id nije obrađen gore (iako bi ga dispatcher trebao uhvatiti)
            return jsonify({"error": f"Model dispatch logic not implemented for {model_id}"}), 500


        best_smiles = result.get("best_smiles")
        best_qed = result.get("best_qed", 0.0)
        model_log = result.get("log", result.get("all_generated_top_10", ""))

        if not best_smiles:
             return jsonify({"error": f"Model {model_id} did not return a 'best_smiles'."}), 500

        generated_mol = Chem.MolFromSmiles(best_smiles)
        if not generated_mol:
            return jsonify({"error": f"Invalid generated SMILES from model: {best_smiles}"}), 500

        similarity_score = 0.0
        if initial_mol:
            fp1 = AllChem.GetMorganFingerprint(initial_mol, 2)
            fp2 = AllChem.GetMorganFingerprint(generated_mol, 2)
            similarity_score = DataStructs.TanimotoSimilarity(fp1, fp2)

        buf2 = BytesIO()
        img2 = Draw.MolToImage(generated_mol, size=(300, 300))
        img2.save(buf2, format="PNG")
        generated_img_str = base64.b64encode(buf2.getvalue()).decode("utf-8")
        generated_mol_block_str = ""
        try:
            AllChem.EmbedMolecule(generated_mol, randomSeed=0xf00d)
            AllChem.UFFOptimizeMolecule(generated_mol)
            generated_mol_block_str = Chem.MolToMolBlock(generated_mol)
        except Exception as e_3d_gen:
            print(f"Error generating 3D for generated molecule: {e_3d_gen}")
            generated_mol_block_str = ""
            
        input_props = {}
        if initial_mol:
            input_props = {
                "input_LogP": Descriptors.MolLogP(initial_mol),
                "input_Molecular Weight": Descriptors.MolWt(initial_mol),
                "input_TPSA": Descriptors.TPSA(initial_mol),
                "input_NumHDonors": Descriptors.NumHDonors(initial_mol),
                "input_NumHAcceptors": Descriptors.NumHAcceptors(initial_mol),
            }

        response_properties = {
            "input_smiles": smiles,
            "input_qed": input_qed,
            "input_image": f"data:image/png;base64,{input_img_str}",
            "input_mol_block": input_mol_block_str,
            **input_props,
            "generated_smiles": best_smiles,
            "generated_qed": best_qed,
            "similarity": similarity_score,
            "generated_image": f"data:image/png;base64,{generated_img_str}",
            "generated_mol_block": generated_mol_block_str,
            "LogP": Descriptors.MolLogP(generated_mol),
            "Molecular Weight": Descriptors.MolWt(generated_mol),
            "TPSA": Descriptors.TPSA(generated_mol),
            "NumHDonors": Descriptors.NumHDonors(generated_mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(generated_mol),
            "model_used": model_id,
            "model_log": str(model_log) 
        }
        return jsonify(response_properties)

    except Exception as e:
        tb_str = traceback.format_exc()
        print("--- FLASK ERROR ---")
        print(tb_str)
        print("--- END FLASK ERROR ---")
        return jsonify({"error": "Internal server error on Flask backend", "details": str(e), "trace": tb_str}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
