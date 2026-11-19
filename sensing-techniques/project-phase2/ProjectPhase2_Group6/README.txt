================================================================================
PROJECT PHASE 2 - STATISTICAL DAMAGE DETECTION
Feature Engineering & Anomaly Detection
================================================================================
Course      : CIV E 789 - Sensing Techniques and Data Analytics, Fall 2026
Instructor  : Mohammad Talebi-Kalaleh, Ph.D., University of Alberta
Group       : 6
Members     : Amilcar Diaz     (CCID: adiazgar)
              Alexander Jordan (CCID: ajordan2)
Submitted   : November 19, 2026

--------------------------------------------------------------------------------
ASSIGNED SENSORS (metadata.sensor_assignments.group_6)
--------------------------------------------------------------------------------
Span 1 (0-20 m)  : S2  (x =   7.5 m), S4  (x =  17.5 m)
Span 2 (20-60 m) : S6  (x =  27.5 m), S9  (x =  42.5 m), S12 (x =  57.5 m)
Span 3 (60-120 m): S15 (x =  72.5 m), S18 (x =  87.5 m), S21 (x = 102.5 m),
                   S24 (x = 117.5 m)
Scenarios: intact, damage1 and damage2, 20 estimates each (10 ambient tests
x 2 segments of 30 s). Features: f1, f2, f3 and MAC1, MAC2, MAC3, with the MAC
computed at the 9 assigned sensors against the mean of the 20 intact mode
shapes.

--------------------------------------------------------------------------------
FILE DESCRIPTIONS
--------------------------------------------------------------------------------
ProjectPhase2_Group6/
  Main.ipynb        Complete analysis notebook: Section 0 (setup), Parts A to
                    F and the note on the use of generative AI. Run top to
                    bottom. It reads phase2_modal_parameters.json and writes
                    every figure to figures/ and every CSV table to results/.
  README.txt        This file.
  phase2_modal_parameters.json
                    Input data provided with the assignment. Not part of the
                    structure requested in the assignment; it is included so
                    the notebook runs as delivered.

  figures/          Every figure the notebook produces, 300 DPI PNG. Nineteen
                    files, one per numbered figure in the notebook.
    part-a/         Figures 1 to 7   - feature distributions
      A1_mode_<k>_freq_dist_3sigma.png        Figures 1 to 3 (modes 1 to 3)
      A2_mode_shapes_3x3.png                  Figure 4
      A3_mode_<k>_mac_distributions.png       Figures 5 to 7 (modes 1 to 3)
    part-b/         Figure 8         - single feature detection
      B2_wasserstein_distance_grouped_bars.png Figure 8
    part-c/         Figures 9 to 13  - bivariate analysis
      C1_f1_f2_scatter.png                    Figure 9
      C1_MAC1_MAC2_scatter.png                Figure 10
      C1_f1_f2_gaussian.png                   Figure 11
      C1_MAC1_MAC2_gaussian.png               Figure 12
      C2_mahalanobis.png                      Figure 13
    part-d/         Figure 14        - feature fusion
      D1_wasserstein_combinations.png         Figure 14
    part-e/         Figures 15 to 18 - principal component analysis
      E1_scree_plot.png                       Figure 15
      E2_PC1_distributions.png                Figure 16
      E3_PC1_PC2_scatter.png                  Figure 17
      E4_wasserstein_k_PCs.png                Figure 18
    part-f/         Figure 19        - comparison of the detection approaches
      F1_method_comparison.png                Figure 19

  results/          Numerical outputs as CSV, one file per result table.
      A3_mac_summary_statistics.csv           Table 6  - MAC mean and std
      B1_univariate_zscore_detection.csv      Table 7  - z-score detection
      B2_wasserstein_distance_summary.csv     Table 8  - W1 and W1/sigma0
      C2_detection_rates.csv                  Table 12 - Mahalanobis rates
      C3_comparison_frequencies.csv           Table 13 - f1, f2 vs (f1, f2)
      C3_comparison_MAC.csv                   Table 14 - MAC1, MAC2 vs pair
      D1_intact_baseline.csv                  Table 15 - standardization
      D1_wasserstein_combinations.csv         Table 16 - 15 combinations
      D1_top3_combinations.csv                Table 17 - top 3 combinations
      D1_best_combination_detection.csv       Table 18 - 4D Mahalanobis rates
      D2_best_features_vs_combinations.csv    Table 19 - features vs combos
      E2_wasserstein_PC1.csv                  Table 20 - PC1 W1/sigma0
      E3_wasserstein_PC1_PC2.csv              Table 21 - PC1-PC2 W
      E3_PC1_PC2_detection.csv                Table 22 - PC1-PC2 rates
      E4_wasserstein_k_PCs.csv                Table 23 - W for k = 1 to 6
      E5_PC1_loadings.csv                     Table 24 - PC1 loadings
      F1_method_comparison.csv                Table 25 - method comparison

--------------------------------------------------------------------------------
HOW TO RUN THIS NOTEBOOK
--------------------------------------------------------------------------------

REQUIREMENTS
    python >= 3.10 with:
        numpy  pandas  scipy (>= 1.13)  matplotlib  scikit-learn  jupyter

        pip install numpy pandas "scipy>=1.13" matplotlib scikit-learn jupyter

    scipy >= 1.13 is needed for scipy.stats.wasserstein_distance_nd (Parts D
    and E).

RUNNING IT
    1. Open Main.ipynb. It looks for phase2_modal_parameters.json in its own
       folder and, if it is not there, in the parent folder.
    2. Kernel -> Restart Kernel and Run All Cells.

    figures/ and results/ are created by the notebook if they do not exist.

--------------------------------------------------------------------------------
NOTES ON THE DATA
--------------------------------------------------------------------------------
* Value i of each natural frequency list and measurement_i of the mode shapes
  are assumed to come from the same 30-s segment, so that Parts C, D and E can
  combine several features of the same measurement into a single point.
* Mode shapes are used as provided (already scaled to a maximum of 1 over the
  24 sensors); the MAC does not depend on scale.
* The notebook checks that there are 20 measurements per scenario, 24 values
  per mode shape and no missing values before any analysis (Section 0.3).
