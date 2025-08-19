# Generative Molecular Design with Reinforcement Learning

<img width="1107" height="409" alt="image" src="https://github.com/user-attachments/assets/a63e3f39-7577-4936-9e60-e6140a9778fe" />


This project is a full-stack web application that demonstrates the use of Artificial Intelligence, specifically Reinforcement Learning (RL), for *de novo* molecular design. The application allows users to input a starting molecule, select a pre-trained AI model, and generate novel molecular structures optimized for specific chemical properties, such as a high "drug-likeness" score (QED).

This project was developed as part of a final thesis under the mentorship of Professor Damir Pintar.

**Live Application Link:** [v0-zavrsni-rad.vercel.app](https://v0-zavrsni-rad.vercel.app/) <!-- Replace with your actual Vercel link if different -->

---

## 🎯 About The Project

Traditional drug discovery is a lengthy, expensive, and often inefficient process, with failure rates exceeding 90% in clinical trials. The chemical space of possible molecules is vast (estimated at 10^60), making exhaustive searches impossible. This project explores how AI can address these challenges by intelligently navigating this space to find promising new molecules.

The core of the project is an RL agent trained to perform a "molecular game": it makes small, iterative changes to a molecule's structure. The agent is rewarded based on the quality of the resulting molecule, learning a strategy to maximize desirable properties over time.

### Key Features

* **Interactive Web Interface:** Users can input a molecule via its SMILES string.
* **Multiple AI Models:** The application hosts several RL models, each representing a different stage of development and optimization strategy.
* **Comprehensive Visualization:** A side-by-side comparison of the input and generated molecules, featuring:
    * 2D structural diagrams.
    * Interactive 3D molecular models rendered with **3Dmol.js**.
    * A detailed table of key physicochemical properties (QED, SA Score, Similarity, LogP, etc.).
* **Targeted Optimization:** The models are trained to maximize the Quantitative Estimate of Drug-likeness (QED) while maintaining structural similarity to the starting molecule.

---

## 🛠️ Technology Stack

This project is a full-stack application with a decoupled architecture.

### Frontend

* **Framework:** Next.js (with App Router)
* **Language:** React & TypeScript
* **Styling:** Tailwind CSS
* **Animation:** Framer Motion
* **3D Visualization:** 3Dmol.js

### Backend & AI

* **Framework:** Flask
* **Language:** Python
* **AI Core:** PyTorch
* **Cheminformatics:** RDKit
* **Molecular Representation:** SELFIES (for robust generation)

### Deployment

* **Frontend:** Vercel
* **Backend:** Google Cloud Run (as a Docker container)

---

## 🧠 Model Evolution

The project was developed through an iterative process, with each model building upon the last to solve key challenges:

* **Model 1 (Baseline):** Started with a simple REINFORCE algorithm using SMILES strings. This approach suffered from instability and frequently generated chemically invalid molecules.
* **Model 2 (Robustness):** Introduced the **SELFIES** molecular representation, which guarantees 100% chemical validity for generated structures, solving a major flaw of the initial model.
* **Model 3 (Stability & Refinement):** Migrated from the REINFORCE algorithm to a more stable **Actor-Critic** architecture. Also integrated the Synthetic Accessibility Score (SA_Score) into the reward function to guide the agent towards molecules that are easier to synthesize.
* **Model 4 (Targeted Optimization):** Shifted focus from general QED optimization to maximizing similarity with a known drug (Celecoxib), directing the optimization towards a specific biological profile.
* **Model 5 (GRPO & GAE):** Implemented advanced RL techniques, including **Generalized Advantage Estimation (GAE)** for training stability and a **GRPO-inspired** algorithm for more aggressive and innovative exploration of the chemical space.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Node.js & npm
* Python & pip
* Conda (recommended for installing RDKit)

### Installation

1.  **Clone the repo**
    ```sh
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    ```
2.  **Install Frontend Dependencies**
    ```sh
    npm install
    ```
3.  **Setup Python Backend**
    * Create and activate a virtual environment:
        ```sh
        python -m venv venv
        source venv/bin/activate # On Windows use `venv\Scripts\activate`
        ```
    * Install Python packages:
        ```sh
        pip install -r python/requirements.txt
        ```
        *(Note: Installing RDKit via pip can be tricky. Using `conda install -c conda-forge rdkit` is often more reliable.)*

4.  **Run the Application**
    * Start the Flask backend:
        ```sh
        python python/run_model.py
        ```
    * In a separate terminal, start the Next.js frontend:
        ```sh
        npm run dev
        ```
    * Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🔮 Future Work

* **Better Representation:** Transition from molecular fingerprints to **Graph Neural Networks (GNNs)** to capture the full 3D structure and relationships within the molecule.
* **Smarter Actions:** Develop a more sophisticated action space based on real chemical reactions and transformations, rather than simple symbol manipulation.
* **Advanced Similarity:** Replace Tanimoto similarity with pre-trained **QSAR models** or other learned metrics to better predict biological activity.



