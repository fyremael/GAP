# Diagnostic Suite

The diagnostic suite is the heart of the programme. The claim is not merely that the model is accurate; the claim is that the model respects protected structure under depth, rollout, hierarchy, and routing.

## 1. Constraint Leakage

For state `x_l` at layer `l`:

`leak_l = ||A x_l|| / (||x_l|| + eps)`.

Report mean, max, percentile-95, and final-layer leakage.

For sequence rollout:

`leak_t = ||A x_t|| / (||x_t|| + eps)`.

Plot leakage against time. A good model should not accumulate illegal structure.

## 2. Protected Energy Drift

Let `P_K` be the protected projector.

`E_K(l) = ||P_K x_l||^2`

`E_perp(l) = ||P_perp x_l||^2`.

Track whether protected energy is preserved, damped, or polluted.

## 3. Commutator Error

For hierarchy transfer:

`C_R = ||A_c R - R_A A_f||_F / (||A_c R||_F + ||R_A A_f||_F + eps)`.

For chain complex transfer:

`C_d = ||d_c P_k - P_{k-1} d_f||_F / denom`.

For routing:

`C_route = ||R_theta P_K - P_K^expert R_theta||_F / denom`.

These are the direct measurements of compatibility.

## 4. Gap Preservation

For the relevant Laplacian or constraint normal operator:

`L = A^* A`.

Measure:

- smallest nonzero eigenvalue `lambda_min_plus`;
- protected nullity estimate;
- gap ratio `lambda_min_plus / lambda_max`;
- drift of the gap across learned hierarchy.

For nonlinear blocks, estimate the Jacobian action on protected and complement modes.

## 5. Complement Amplification

Estimate local spectral norm on complement:

`sigma_perp = ||P_perp J P_perp||_2`.

Use power iteration with vector-Jacobian products.

Also estimate nonnormal amplification using short-horizon perturbation growth:

`amp_k = ||J^k v|| / ||v||`.

This matters because attention/MoE blocks may be stable by eigenvalues but unstable by pseudospectrum.

## 6. Rollout Stability

For dynamical tasks:

- prediction error vs time;
- constraint leakage vs time;
- energy/enstrophy/mass drift where applicable;
- divergence/curl/harmonic component drift;
- catastrophic error onset time.

## 7. Resolution Transfer

Train on one resolution; test on finer/coarser resolutions.

Metrics:

- error under resolution change;
- leakage under resolution change;
- commutator error for learned or fixed transfer maps;
- gap consistency across scales.

## 8. Interpretability Ledger

At each layer, log:

- protected component norm;
- learned complement norm;
- cross-leakage;
- attention mass by component;
- routing distribution by component;
- worst offending layer/expert.

This produces a mode-level audit trail.
