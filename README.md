#Temporal GNN for Bitcoin AML
This project implements a sophisticated graph neural network (GNN) pipeline to detect illicit (AML) transactions in the Elliptic Bitcoin dataset.
The core challenge of this dataset is severe concept drift: the behavior of illicit and licit entities changes significantly over time. This model is designed to adapt to this drift using a multi-stage approach that includes temporal encoding, contrastive pre-training, and an iterative self-training loop with domain-adaptation techniques.
🚀 Key Features
•	Temporal Graph Encoder: Uses a GINConv (Graph Isomorphism Network) coupled with a GRU (Gated Recurrent Unit) to learn node embeddings that evolve over time.
•	Contrastive Pre-training: The temporal encoder is first pre-trained on all data (labeled and unlabeled) using an NT-Xent contrastive loss to learn robust representations of transaction behavior.
•	Iterative Adaptation Pipeline: The model is fine-tuned in rounds, with each round designed to adapt to data drift:
o	Importance Weighting: A domain classifier is trained to distinguish "early" (t≤30) from "later" (t=31-34) data. The resulting weights are used to up-weight training samples that resemble the future, non-stationary data.
o	Gated Self-Training: The model fine-tunes its head on high-confidence pseudo-labels from the validation timesteps (t=31-34). This update is "gated," meaning it is only kept if it improves the validation F2-score.
o	Consistency-Based Pseudo-Labeling: High-confidence, consistency-checked predictions on the training timesteps (t≤30) are added to the label set, expanding the training data for the next round.
•	Recall-Focused Optimization: Uses Focal Loss to handle the severe class imbalance and optimizes for the F2-score of the illicit class, prioritizing recall.
•	Test-Time Adaptation (TTA): At inference time, the model automatically chooses the best strategy:
1.	Fixed Threshold: A single, globally-tuned threshold ($\tau$).
2.	Adaptive TTA: Employs per-timestep adaptive thresholds ($\tau_t$) based on a target illicit rate and uses an EM algorithm to correct for prior probability shift ($\pi_t$).
🏛️ Model Architecture
The model is composed of two main components:
1.	TemporalGINEncoder (Encoder):
o	This module's job is to create a "memory" for each node.
o	At each timestep $t$, it takes the node's static features ($x$) and its previous hidden state ($h_{t-1}$) from the GRU.
o	It performs graph convolution using GINConv on the cumulative subgraph up to time $t$.
o	The result is passed back into the GRU to produce the new temporal embedding, $h_t$.
o	This $h_t$ captures the node's behavior and neighborhood context up to that point in time.
2.	GATClassifier (Classifier):
o	This module makes the final licit/illicit prediction.
o	It uses the powerful GATv2Conv (Graph Attention Network v2).
o	Its input is a concatenation of the node's original static features ($x$) and its final temporal embedding ($h_t$) from the encoder.
o	This allows the classifier to consider both the node's intrinsic properties and its complex temporal behavior.
🔄 Training & Adaptation Pipeline
The model is trained using a sophisticated strategy to combat concept drift.
1. Pre-training (Unsupervised)
Before any classification, the TemporalGINEncoder is trained on all 49 timesteps using a contrastive learning task.
•	Goal: Teach the encoder what "similar" temporal transaction patterns look like, regardless of their label.
•	Method: For a given node, two augmented "views" are created (using feature dropout, noise, and edge dropping). The model is trained to pull the representations of these two views (anchor, positive) closer together while pushing them away from all other nodes (negatives) in the batch.
•	Loss: NT-Xent Loss.
2. Iterative Fine-Tuning (Supervised)
The model then enters a loop of fine-tuning and adaptation (e.g., for 3 rounds).
For each round $r$:
1.	Train Downstream: The GATClassifier and the pre-trained TemporalGINEncoder are trained on the labeled nodes (t≤30) to predict illicit/licit.
o	Loss: FocalLoss to prioritize the rare illicit class.
o	Adaptation: If enabled, ImportanceWeighting is used to focus the loss on train samples that look most like the validation data.
o	Stability: An EMA (Exponential Moving Average) of the classifier's weights is maintained.
2.	Calibrate & Evaluate (on Val, t=31-34):
o	The model's probability outputs are calibrated using Temperature Scaling (T).
o	The best classification threshold ($\tau$) is found by optimizing the F2-score on the validation set.
3.	Self-Train on Validation (Gated):
o	The calibrated model makes predictions on unknown nodes in the validation set (t=31-34).
o	The classifier head is briefly fine-tuned on these new, high-confidence pseudo-labels.
o	Gating: This new "self-trained" model is only kept if its F2-score on the validation set is better than the model from step 2. If not, it's discarded.
4.	Pseudo-Label on Training:
o	The chosen model (from step 2 or 3) is used to make high-confidence, consistency-checked predictions on unknown nodes in the training set (t≤30).
o	These new labels are permanently added to the labeled_data object.
This loop repeats, allowing the model to iteratively refine its understanding and expand its training set with high-quality, self-generated labels.
3. Final Evaluation (on Test, t>34)
After the final round, the best-performing model is loaded and evaluated on the unseen test set (timesteps 35-49). It compares two strategies and prints the results for both:
•	[A] Adaptive Strategy: Uses prior probability correction and per-timestep adaptive thresholds.
•	[B] Fixed Strategy: Uses the single, global threshold $\tau$ found on the validation set.
The model reports the metrics for both and selects the one with the higher F2-score as the final solution.
⚙️ Setup and Execution
1. Environment Setup
This project requires PyTorch and several PyTorch Geometric (PyG) libraries. The specific pip commands for PyG are version-dependent on your local PyTorch and CUDA versions. The script attempts to auto-detect this.
Bash
# 1. Install PyTorch (e.g., from https://pytorch.org/)
# (Example for CUDA 12.1)
pip install torch torchvision torchaudio

# 2. Install PyG and dependencies
# The script handles this, but you can also do it manually.
# Find your specific {TORCH} and {CUDA} versions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip install torch-geometric pyg-lib
2. Dataset
1.	Download the Elliptic Bitcoin Dataset from Kaggle.
2.	Place the .csv files (elliptic_txs_features.csv, elliptic_txs_edgelist.csv, elliptic_txs_classes.csv) in the /kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/ directory relative to the script, or update the paths in the BitcoinAML.py file:
Python
# ==== Paths ====================================================================
features_path = 'path/to/elliptic_txs_features.csv'
edges_path    = 'path/to/elliptic_txs_edgelist.csv'
classes_path  = 'path/to/elliptic_txs_classes.csv'
3. Run the Pipeline
To run the full training and evaluation pipeline, simply execute the Python script:
Bash
python BitcoinAML.py

