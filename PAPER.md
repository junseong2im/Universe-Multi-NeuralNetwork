# UMN: Universe Multi Neural Network for Olfactory Mixture Similarity Prediction via Differentiable N-body Simulation

## Abstract

Predicting the perceptual similarity of olfactory mixtures from their molecular composition remains an open problem in computational olfaction. Existing approaches either rely on hand-crafted physicochemical descriptors with manual feature selection or employ graph neural networks that operate on individual molecules without modeling inter-molecular interactions at the mixture level. In this work, we propose Universe Multi Neural network (UMN), a physics-simulation-based architecture that maps molecular fingerprints to celestial bodies with learnable mass, position, and velocity, performs differentiable N-body gravitational simulation, and extracts orbital stability features as mixture-level representations. We evaluate UMN on the Snitz et al. (2013) mixture similarity dataset (360 pairs) using 5-seed x 5-fold cross-validation. UMN achieves a Pearson correlation of r = 0.780 (std = 0.039) without any manual feature engineering, compared to r >= 0.49 for a non-optimized descriptor-based model. Through systematic experiments across 9 versions (v17-v25), we demonstrate that (1) the physics engine architecture is near-optimal in its original form, (2) multi-restart training is the most effective optimization strategy, and (3) contrastive learning, physics-based loss functions, and architectural modifications all degrade performance on this small-data regime. These findings highlight the effectiveness of physics-inspired inductive biases for modeling molecular interactions and the importance of matching optimization strategies to loss landscape geometry.

## 1. Introduction

The human olfactory system can discriminate an enormous number of volatile molecules. Bushdid et al. (2014) estimated that humans can distinguish at least one trillion olfactory stimuli. Despite this remarkable discriminatory capacity, computational prediction of olfactory perception from molecular structure remains challenging, particularly at the mixture level where multiple odorants interact to produce a unified percept.

Prior computational approaches to olfactory prediction have primarily focused on individual molecule properties. The DREAM Olfaction Prediction Challenge (Keller et al., 2017) established benchmarks for predicting single-molecule perceptual attributes from chemical features. More recently, the Principal Odor Map (POM) by Lee et al. (2023) employed graph neural networks (GNN) to predict odor descriptors for individual molecules, achieving AUROC of 0.79 across 55 descriptors. While POM represents significant progress in single-molecule olfaction, it does not directly model mixture-level interactions where the combined percept of multiple molecules differs from the sum of individual percepts.

For mixture similarity prediction, Snitz et al. (2013) proposed a model based on physicochemical descriptors. Their optimized model, which selected 21 descriptors from an initial pool of 1,433, achieved a Pearson correlation of r = 0.85 on a train/test split. However, this approach requires extensive domain-specific feature engineering and descriptor selection, limiting its applicability to new domains.

A fundamental challenge in mixture modeling is capturing inter-molecular interactions. Standard approaches represent mixtures as averaged molecular features, which discards interaction information. Graph neural networks (Gilmer et al., 2017) and attention mechanisms can model pairwise interactions but require explicit interaction graphs or quadratic computation. In the broader machine learning literature, physics-based neural networks have demonstrated that incorporating physical priors can improve data efficiency that is particularly relevant for the small datasets typical in olfaction research. Hamiltonian Neural Networks (Greydanus et al., 2019) structurally enforce energy conservation, while Neural Relational Inference (Kipf et al., 2018) learns interaction dynamics in N-body systems.

In this work, we propose Universe Multi Neural network (UMN), a novel architecture that bridges molecular fingerprints and mixture-level representation through differentiable gravitational simulation. UMN converts each molecule in a mixture into a celestial body with learnable physical properties (mass, position, velocity), simulates their gravitational interactions via N-body dynamics, and extracts orbital stability features as the mixture representation. This approach naturally handles variable-size mixtures, is permutation invariant, and encodes pairwise molecular interactions through the simulation dynamics rather than explicit attention or message-passing.

Our contributions are as follows:

(a) We propose UMN, a differentiable N-body simulation architecture for mixture-level molecular representation learning that requires no manual feature engineering.

(b) We conduct systematic experiments across 9 model versions, evaluating contrastive learning, physics-based loss functions, architectural modifications, and training strategies on the mixture similarity prediction task.

(c) We demonstrate that multi-restart training is the dominant factor in performance improvement for physics-simulation models with multi-modal loss landscapes, outperforming recent techniques such as Stochastic Weight Averaging and contrastive learning.

(d) We provide detailed analysis including statistical significance tests, external baseline comparisons, embedding visualizations, and error analysis.

## 2. Related Work

### 2.1 Molecular Representation Learning

Classical molecular representations include ECFP/Morgan fingerprints (Rogers and Hahn, 2010), which encode circular substructures as fixed-length binary vectors. Neural approaches include message-passing neural networks (Gilmer et al., 2017) that learn molecular representations from atomic graphs, and transformer-based models that treat SMILES strings as sequences. While these methods achieve state-of-the-art performance on single-molecule property prediction, extending them to mixtures requires aggregation (e.g., mean pooling) that discards inter-molecular interaction information.

### 2.2 Olfactory Prediction

Computational olfaction has progressed from expert-curated descriptor models to learned representations. Snitz et al. (2013) demonstrated that perceptual similarity of odor mixtures can be predicted from physicochemical descriptors with r = 0.85, though this required selecting 21 descriptors from 1,433 candidates. Keller et al. (2017) organized the DREAM Olfaction Prediction Challenge for single-molecule perception prediction. Lee et al. (2023) developed the Principal Odor Map using GNN-based molecular embeddings, achieving AUROC of 0.79 across 55 odor descriptors on single molecules. However, the mixture similarity prediction problem that requires modeling how multiple molecules combine to form a unified percept is less studied.

### 2.3 Physics-Informed Neural Networks

Incorporating physical priors into neural networks has shown benefits in data-limited regimes. Hamiltonian Neural Networks (HNN; Greydanus et al., 2019) parameterize the Hamiltonian and learn dynamics that conserve energy by construction. Physics-Informed Neural Networks (PINN; Raissi et al., 2019) add PDE residuals as regularization terms. Neural ODE (Chen et al., 2018) treats neural network evaluation as continuous-time dynamics. Neural Relational Inference (NRI; Kipf et al., 2018) learns to infer interactions in multi-body systems. Unlike these approaches, UMN uses gravitational simulation as a feature extractor rather than as a learning target and does not enforce physical law compliance. We note that in our experiments (Section 5.3), explicitly enforcing physical law compliance through HNN and PINN-style losses degraded prediction performance.

## 3. Method

### 3.1 Problem Formulation

Given two olfactory mixtures A = {a_1, ..., a_m} and B = {b_1, ..., b_n}, where each a_i and b_j is a molecule represented by its Morgan fingerprint x in {0, 1}^2048, the task is to predict their perceptual similarity s in [0, 1].

### 3.2 Architecture Overview

UMN consists of three modules: (1) InputHardwareLayer that transforms molecular fingerprints into internal representations, (2) PhysicsProcessingEngine that maps representations to physical quantities and simulates gravitational dynamics, and (3) SimilarityHead that predicts pairwise similarity from physics embeddings. Fig. 1 illustrates the full pipeline.

![Fig. 1. UMN Architecture](figures/architecture.png)

### 3.3 Module 1: InputHardwareLayer

Each molecular fingerprint x in {0, 1}^2048 is transformed into an atom vector c in R^128 through three stages:

Grid Mapping: The 2048-dimensional fingerprint is reshaped into an 8 x 16 grid, imposing spatial structure on the binary features.

Sparse Activation: Each grid cell is passed through a learnable activation function with sparsity regularization (L_sparsity = lambda * ||activation||_1, lambda = 0.01) to produce a sparse constellation pattern.

Channel Transformation: The grid pattern is projected to a 128-dimensional atom vector via a linear layer followed by layer normalization.

For a mixture of m molecules, this produces a tensor C in R^(m x 128) with an associated mask M in {0, 1}^m indicating valid molecular slots (maximum 20 molecules per mixture).

### 3.4 Module 2: PhysicsProcessingEngine

#### 3.4.1 ConstellationToCelestial Mapping

Each atom vector c_i in R^128 is mapped to physical quantities through three separate linear projections:

m_i = clamp(softplus(W_m * c_i + b_m), max=5.0)    where W_m in R^(128 x 1)
p_i = 2.0 * tanh(W_p * c_i + b_p)                   where W_p in R^(128 x 3)
v_i = 0.5 * tanh(W_v * c_i + b_v)                   where W_v in R^(128 x 3)

where m_i in R^1 is mass (strictly positive via Softplus and bounded by clamp), p_i in R^3 is position (bounded to [-2, 2]), and v_i in R^3 is velocity (bounded to [-0.5, 0.5]).

We note that each projection is deliberately kept as a single linear layer. Experiments with multi-layer perceptrons (Section 5.4) showed that nonlinear mappings degrade performance, likely because linear projections preserve clean gradient flow through the subsequent simulation.

#### 3.4.2 GravitationalEngine

The engine performs T-step N-body simulation using Verlet integration with differentiable operations. At each time step t:

d_ij = ||r_j(t) - r_i(t)||,  clamped to d_ij >= epsilon

a_i(t) = G * sum_{j != i} [ m_j * (r_j(t) - r_i(t)) / d_ij^3 ]

v_i(t+1) = v_i(t) + clamp(a_i(t), -100, 100) * dt

r_i(t+1) = r_i(t) + v_i(t+1) * dt

where G = clamp(exp(log_G), 0.01, 10.0) is a learnable gravitational constant (log_G is an nn.Parameter), epsilon = 1e-4 is a distance clamping threshold preventing singularities, and dt = 0.05 is the time step. The trajectory tensor H in R^(B x T x N x 3) records all positions across time steps.

Mass decay is applied at each step to prevent numerical instability in long simulations. The optimal simulation length was found to be T = 4 steps; longer simulations (T >= 8) increase numerical instability and degrade performance (Section 5.5).

#### 3.4.3 OrbitalStabilityEvaluator

The evaluator extracts a 20-dimensional physics embedding e in R^20 from the trajectory tensor H:

Orbital stability score (1D): Spectral analysis of position variance over time, measuring how stable the system is.

Spectral fingerprint (8D): Eigenvalues of the interaction matrix at the final time step, capturing the geometric structure of the molecular arrangement.

Mean mass, velocity, angular momentum (3D): Aggregate physical quantities of the system.

Kinetic/potential energy ratio (2D): Energy distribution between motion and configuration.

Trajectory variability metrics (6D): Variance of inter-molecular distances, pairwise distance statistics, and trajectory smoothness.

### 3.5 Module 3: SimilarityHead

Given physics embeddings e_A and e_B for two mixtures, similarity is predicted as:

p_A = Proj(e_A),  p_B = Proj(e_B)    where Proj: R^20 -> R^64 (Linear + LayerNorm + GELU)

s_hat = sigma(MLP(|p_A - p_B|))     where MLP: R^64 -> R^32 -> R^1, sigma = sigmoid

The model is trained with MSE loss plus sparsity regularization:

L = (1/N) * sum_{i=1}^{N} (s_hat_i - s_i)^2 + lambda * L_sparsity

where s_i is the normalized ground truth similarity in [0, 1] and lambda = 0.01.

### 3.6 Multi-Restart Training

The loss landscape of UMN contains multiple local minima due to the complex interaction between the learnable mapping, gravitational dynamics, and prediction head. We address this with multi-restart training:

(1) For each data fold, train N independent models with different random seeds.
(2) Select the model with the highest validation Pearson r from each restart.
(3) Return the best model across all N restarts.

Increasing restarts from 3 to 10 improved mean r from 0.680 to 0.780, a 14.7% improvement that exceeded all other optimization techniques tested (Section 5.5).

## 4. Experimental Setup

### 4.1 Dataset

We use the Snitz et al. (2013) mixture similarity dataset, which contains 360 odorant mixture pairs with perceptual similarity ratings from 139 human subjects. Each mixture contains 1 to 43 molecular components. Similarity ratings range from 0 (completely different) to 100 (identical).

### 4.2 Evaluation Protocol

All experiments use 5-seed x 5-fold cross-validation (25 evaluations total), reporting mean and standard deviation of Pearson correlation coefficient r. Statistical significance is assessed using paired t-tests and Wilcoxon signed-rank tests across the 25 fold-level results.

### 4.3 Training Details

Optimizer: Adam (lr = 3e-4, weight_decay = 1e-4). Scheduler: CosineAnnealingLR. Batch size: 16. Epochs: 60 with early stopping (patience = 15). Gradient clipping: max_norm = 1.0. Restarts: 10 (best selection).

### 4.4 Baselines

We compare against the following baselines:

(a) Snitz et al. (2013) Optimized: Feature-engineered model selecting 21 from 1,433 physicochemical descriptors, evaluated on train/test split (r = 0.85).

(b) Snitz et al. (2013) Simple: Single structural vector representation without feature selection, evaluated on train/test split (r >= 0.49).

(c) Tanimoto Similarity: Average pairwise Tanimoto coefficient between Morgan fingerprints of mixture components.

(d) Component Count Ratio: min(|A|, |B|) / max(|A|, |B|) as a size-based heuristic.

(e) Random/Mean Prediction: Uniform random or constant mean prediction as lower bounds.

## 5. Results

### 5.1 Main Results

Table 1 presents the comparison of UMN against baselines and ablation variants.

Table 1. Mixture Similarity Prediction Performance

| Method | Pearson r | Std | Evaluation |
|--------|:---------:|:---:|:----------:|
| Snitz et al. (2013) Optimized | 0.85 | - | Train/Test |
| UMN v25 (10-restart) | 0.780 | 0.039 | 5s x 5f CV |
| UMN v25-C (HP search) | 0.732 | 0.038 | 5s x 5f CV |
| UMN v25-A (SWA + warmup) | 0.727 | 0.037 | 5s x 5f CV |
| UMN v18 (3-restart) | 0.680 | - | 5s x 5f CV |
| UMN v19 (+InfoNCE) | 0.594 | 0.085 | 5s x 5f CV |
| Snitz et al. (2013) Simple | >= 0.49 | - | Train/Test |
| Tanimoto Similarity | 0.439 | - | Full data |
| Component Count Ratio | 0.417 | - | Full data |
| UMN v25-B (+Bushdid data) | 0.565 | 0.073 | 5s x 5f CV |
| UMN v21 (+6 enhancements) | 0.553 | 0.117 | 5s x 5f CV |
| UMN v22 (+HNN/PINN loss) | 0.532 | 0.119 | 5s x 5f CV |
| UMN v23 (+Multi-scale) | 0.520 | 0.112 | 5s x 5f CV |
| UMN v24 (+MLP mapper) | 0.440 | 0.090 | 5s x 5f CV |
| UMN v20 (+Triplet loss) | 0.436 | 0.112 | 5s x 5f CV |
| Random Prediction | ~0.0 | - | - |

UMN achieves r = 0.780 without any manual feature engineering, compared to Snitz et al.'s feature-engineered r = 0.85. When compared to their non-optimized model (r >= 0.49) which similarly requires no feature selection, UMN represents a 59% improvement. Compared to the chemical structure baseline (Tanimoto, r = 0.439), UMN achieves a 77.6% higher correlation, demonstrating that physics-simulation-based representations capture perceptual information beyond chemical similarity.

We note that UMN and Snitz et al. (2013) use different evaluation protocols. Snitz et al. report performance on a single train/test split with optimized feature selection performed on the full data. UMN is evaluated under the stricter 5-seed x 5-fold cross-validation protocol where no information from test folds is used during training or model selection.

![Fig. 2. Experimental Results Comparison](figures/experiment_results.png)

### 5.2 Statistical Significance

Table 2 presents paired t-test and Wilcoxon signed-rank test results across 25 folds comparing UMN v25 baseline against optimization strategies.

Table 2. Statistical Significance Tests (25 folds)

| Comparison | Mean diff | t-stat | p-value (t) | p-value (W) |
|------------|:---------:|:------:|:-----------:|:-----------:|
| Baseline vs A (SWA) | +0.053 | 7.85 | 4.4e-08 | 2.6e-06 |
| Baseline vs B (Data) | +0.216 | 16.12 | 2.3e-14 | 6.0e-08 |
| Baseline vs C (HP) | +0.048 | 5.83 | 5.2e-06 | 1.0e-05 |
| A (SWA) vs C (HP) | -0.004 | -0.64 | 0.529 | 0.339 |

The baseline (10-restart with original training) is statistically significantly better than all three strategies (p < 0.001 for both tests). Strategies A (SWA + warmup) and C (hyperparameter search) show no significant difference between each other (p = 0.529).

### 5.3 Contrastive Learning and Physics-Based Losses (v19-v22)

We tested two categories of auxiliary objectives:

Contrastive Learning (v19-v21): Adding InfoNCE (v19, r = 0.594) or Triplet Margin Loss (v20, r = 0.436) degraded performance. The contrastive objective (making same-mixture representations similar) conflicts with the similarity regression objective (predicting continuous similarity between different mixtures). Additional techniques including adaptive margin, hard negative mining, and similarity-weighted loss (v21, r = 0.553) partially recovered but did not exceed baseline. The small dataset (360 pairs) is fundamentally insufficient for contrastive learning, which typically requires thousands of samples.

Physics-Based Losses (v22): We applied Hamiltonian trajectory matching, PINN regularization (enforcing F = ma residuals), and spectral matching simultaneously. Performance dropped to r = 0.532 with high variance (std = 0.119). The physics losses dominated gradients and interfered with the primary MSE objective. This finding suggests that physical correctness of the simulation is orthogonal to its discriminative power for similarity prediction.

### 5.4 Architectural Modifications (v23-v24)

Capacity Expansion (v23): Adding multi-scale simulation (T = 2, 4, 8 simultaneously), trajectory attention, and 8D latent space yielded r = 0.520. While top folds exceeded baseline (0.709), bottom folds collapsed (0.282), indicating that added complexity increases the number of poor local minima.

Internal Structure Changes (v24): Replacing the linear ConstellationToCelestial mapper with a 2-layer MLP (+ LayerNorm + residual) and expanding the physics embedding from 20D to 32D resulted in r = 0.440. The linear mapper's clean gradient flow is essential for propagating learning signals through the simulation.

### 5.5 Training Strategy Optimization (v25)

We fixed the architecture and tested three optimization strategies:

Strategy A (Training optimization): 10-restart, SWA, warmup (5 epochs), cosine decay, gradient accumulation (effective batch size 64). Result: r = 0.727.

Strategy B (Data augmentation): Converting Bushdid et al. (2014) discrimination data to pseudo-similarity via Jaccard index, adding 1,046 training pairs. Result: r = 0.565. The Bushdid data measures discriminability rather than perceptual similarity, introducing domain shift that degraded performance.

Strategy C (Hyperparameter search): Grid search over simulation length T and time step dt (7 combinations). T = 2 (r = 0.729) and T = 4 (r = 0.729) performed equally. T >= 8 showed sharp degradation (r < 0.461). Result: r = 0.732.

Baseline (original training + 10-restart): r = 0.780. The simplest approach of merely increasing restart count from 3 to 10 outperformed all sophisticated optimization strategies, suggesting that the loss landscape is multi-modal and that broadening the search over initializations is more effective than refining gradient-based optimization.

### 5.6 Cross-Validation Distribution

![Fig. 3. 25-fold Distribution](figures/boxplot_fold_distribution.png)

Fig. 3 shows the distribution of Pearson r across 25 folds for each strategy. The baseline (10-restart) shows both the highest median and the smallest interquartile range, confirming its superiority in both mean performance and stability.

### 5.7 Error Analysis

![Fig. 4. Error Analysis](figures/error_analysis.png)

Fig. 4 (left) shows predicted versus true similarity. The model tends to underpredict extremely high similarities (true = 100, identical mixtures), with prediction errors up to 44 points. Mid-range similarities (40-60) are predicted with the highest accuracy (mean absolute error < 5). Fig. 4 (right) shows that prediction error does not systematically vary with mixture complexity (total number of components), indicating that the model handles variable-size mixtures consistently.

### 5.8 Embedding Visualization

![Fig. 5. t-SNE of Physics Embeddings](figures/tsne_embeddings.png)

Fig. 5 shows t-SNE projection of the 20D physics embeddings colored by pair similarity. Embeddings of highly similar mixture pairs tend to cluster spatially, while dissimilar pairs are distributed more broadly, confirming that the physics simulation produces discriminative representations.

### 5.9 Model Efficiency

Table 3. Model Efficiency

| Component | Parameters | Proportion |
|-----------|:----------:|:----------:|
| InputHardwareLayer | 279,045 | 97.0% |
| PhysicsProcessingEngine | 911 | 0.3% |
| Projection + SimilarityHead | 7,745 | 2.7% |
| Total | 287,701 | 100% |

Inference time: 0.67 ms per pair (batch size 16, single GPU). The physics engine itself uses only 911 parameters (primarily the learnable gravitational constant log_G), demonstrating that the simulation dynamics are controlled by a minimal set of learnable parameters while the InputHardwareLayer bears the representational burden.

## 6. Discussion

### 6.1 Why Gravitational Simulation Works

The effectiveness of UMN on the mixture similarity task can be attributed to several properties of N-body gravitational simulation:

Permutation Invariance: The simulation output is invariant to the ordering of molecules, a natural requirement for mixture representation.

Interaction Modeling: Gravitational forces between all pairs of bodies implicitly model pairwise molecular interactions without explicit attention mechanisms or message-passing steps.

Physical Inductive Bias: The simulation generates trajectories from which physically meaningful features (stability, energy, angular momentum) are automatically extracted. These features provide a structured representation space compared to arbitrary learned features.

The mathematical structure of gravitational force (inverse-square law) is isomorphic to van der Waals interactions between molecules, providing a physically motivated basis for the analogy.

### 6.2 Linear Mapper Optimality

A surprising finding is that the linear ConstellationToCelestial mapper outperforms deeper alternatives. This can be understood through the gradient flow perspective: the physics simulation already introduces significant nonlinearity through the N-body dynamics. Adding nonlinear mapping before the simulation creates compound nonlinearity that makes optimization difficult. The linear layer provides a clean gradient pathway from the loss through the simulation to the input layer.

### 6.3 Multi-Restart as Loss Landscape Navigation

The dominance of multi-restart training (r = 0.780) over sophisticated optimization techniques like SWA (r = 0.727) suggests that the UMN loss landscape has multiple distinct basins of attraction. SWA averages model weights across training, which can be counterproductive when different restarts converge to qualitatively different solutions. In contrast, multi-restart directly increases the probability of finding a high-quality basin.

### 6.4 Small Data Regime Considerations

The Snitz dataset (360 pairs) represents a small-data regime where techniques designed for large datasets (contrastive learning, data augmentation) can be counterproductive. Contrastive learning requires sufficient negative samples that are unavailable with 360 pairs. Domain-shifted augmentation (Bushdid data) introduces noise rather than useful signal. These findings suggest that for small datasets, simple architectures with appropriate optimization (multi-restart) outperform complex architectures with sophisticated training.

## 7. Conclusion

We presented Universe Multi Neural network (UMN), a differentiable N-body gravitational simulation architecture for olfactory mixture similarity prediction. UMN achieves r = 0.780 on the Snitz mixture similarity dataset using 5-seed x 5-fold cross-validation without manual feature engineering. Through systematic experimentation across 9 model versions, we identified that (1) the physics engine architecture is near-optimal in its simplest form, (2) multi-restart training is the most effective optimization strategy for the multi-modal loss landscape, (3) contrastive learning and physics-based loss functions are counterproductive in this small-data regime, and (4) external data augmentation from mismatched domains degrades performance.

Limitations of this work include the small dataset size (360 pairs), the absence of direct comparison using identical evaluation protocols with Snitz et al. (2013), and the lack of chemical interpretability of the learned physical quantities (mass, trajectory, orbital stability). The multi-restart dependency suggests that fundamental improvements in training stability are needed.

Future directions include constructing larger olfactory mixture datasets, interpreting the learned physical quantities in chemical terms, ensemble methods combining simulation-based and GNN-based features, and extending the model to incorporate concentration information.

## References

[1] Bushdid, C., Magnasco, M. O., Vosshall, L. B., and Keller, A. (2014). Humans can discriminate more than 1 trillion olfactory stimuli. Science, 343(6177), 1370-1372.

[2] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. (2018). Neural ordinary differential equations. In Advances in Neural Information Processing Systems 31 (NeurIPS 2018), pp. 6571-6583.

[3] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. (2017). Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning (ICML 2017), pp. 1263-1272.

[4] Greydanus, S., Dzamba, M., and Yosinski, J. (2019). Hamiltonian neural networks. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 8026-8037.

[5] Keller, A., Gerkin, R. C., Guan, Y., Dhurandhar, A., Turu, G., Szalai, B., et al. (2017). Predicting human olfactory perception from chemical features of odor molecules. Science, 355(6327), 820-826.

[6] Kipf, T., Fetaya, E., Wang, K.-C., Welling, M., and Zemel, R. (2018). Neural relational inference for interacting systems. In Proceedings of the 35th International Conference on Machine Learning (ICML 2018), pp. 2688-2697.

[7] Lee, B. K., Mayhew, E. J., Sanchez-Lengeling, B., Wei, J. N., Qian, W. W., Little, K. A., et al. (2023). A principal odor map unifies diverse tasks in human olfaction. Science, 381(6661), 999-1006.

[8] Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

[9] Ravia, A., Snitz, K., Honigstein, D., Finkel, M., Zirler, R., Perl, O., et al. (2020). A measure of smell enables the creation of olfactory metamers. Nature, 588(7836), 118-123.

[10] Rogers, D. and Hahn, M. (2010). Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5), 742-754.

[11] Snitz, K., Yablonka, A., Weiss, T., Frumin, I., Khan, R. M., and Sobel, N. (2013). Predicting odor perceptual similarity from odor structure. PLoS Computational Biology, 9(9), e1003184.

[12] Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., et al. (2019). Analyzing learned molecular representations for property prediction. Journal of Chemical Information and Modeling, 59(8), 3370-3388.
