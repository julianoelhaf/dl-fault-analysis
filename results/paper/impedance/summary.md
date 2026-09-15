# Impedance-based fault-location baseline (true parameters)

Classical physically grounded baselines evaluated on the 90 kV double-line 50 ms FL data with the **true per-episode line parameters** (R'/X', length, sequence impedances) from the scenario metadata. These parameters are NOT available to the data-driven models.

- Window tensor: (207506, 320, 48); samples/cycle at 50 Hz: 128
- Distance error in **% of line length** (= MAE units of the DL FL table).

## Headline (double-ended synchronized, true parameters)

- per_window (same fault_start windows as DL): MAE 3.62% · median 0.54% · <1% in 60% of windows
- per_episode (developed window): MAE 5.08% · median 0.028% · <1% in 86% of episodes

## Single-ended reactance (one terminal)

- per_episode mean-side MAE 217.3% · median 17.27% (unstable tail under load/infeed/Rf -> one terminal is insufficient)

## Aggregate table

   protocol             method     n        mae    median        rmse        p90         p99  hit_1pct  hit_3pct  hit_5pct
 per_window       double_ended 81030   3.623930  0.539122   21.647698   6.972626   41.445210  0.601678  0.785981  0.859867
 per_window  single_ended_mean 81198 223.597235 37.965790  760.836333 538.766552 2748.919356  0.041553  0.133119  0.199389
 per_window single_ended_worst 81198 389.265992 62.077338 1414.153252 905.887697 4892.629462  0.026183  0.092786  0.144979
per_episode       double_ended  8787   5.078334  0.028452   33.795769   7.806273   74.752997  0.860931  0.878798  0.890292
per_episode  single_ended_mean  9022 217.318218 17.274576  854.838720 456.578798 3430.121443  0.104411  0.226890  0.302039
per_episode single_ended_worst  9022 314.490574 28.792537 1233.101055 696.676045 4447.098263  0.076480  0.175238  0.236644


## Per-line (double-ended, per_episode)

      line      n      mae   median      rmse       p90       p99  hit_1pct  hit_3pct  hit_5pct
Line_1_2_a 3057.0 4.430082 0.024233 30.785363  4.502925 60.262928  0.871443  0.889434  0.902519
Line_1_2_b 1284.0 5.319917 0.022304 35.498520  6.229461 79.838860  0.859034  0.880841  0.893302
Line_2_3_a 3118.0 5.746475 0.035186 40.060943 10.572117 76.282954  0.856639  0.873637  0.882938
Line_2_3_b 1328.0 4.768281 0.032047 19.727642 12.546745 85.026501  0.848645  0.864458  0.876506


## Double-ended error sensitivity (per_episode)

              factor        bucket    n      mae   median      rmse       p90        p99  hit_1pct  hit_3pct  hit_5pct
fault_resistance_ohm (-0.001, 1.0]  772 5.518441 0.035579 37.782000  6.745927  96.834202  0.865285  0.882124  0.888601
fault_resistance_ohm    (1.0, 3.0] 1786 4.908980 0.029990 29.464838  9.035726  75.852536  0.852184  0.871781  0.885218
fault_resistance_ohm    (3.0, 6.0] 2664 5.475758 0.027033 40.192418  5.430948  79.230456  0.869369  0.887763  0.898273
fault_resistance_ohm   (6.0, 10.0] 3565 4.770890 0.027501 29.393221  8.636415  64.567455  0.858065  0.874895  0.887237
       true_distance    near(<20%) 1807 5.132187 0.028872 23.549694  7.855103  59.036239  0.867737  0.885445  0.894300
       true_distance           mid 5249 4.243456 0.020262 32.456476  6.000092  67.955593  0.861878  0.880739  0.894456
       true_distance     far(>80%) 1731 7.553757 0.064297 44.995345 27.331649 116.011826  0.850953  0.865973  0.873484
    n_faulted_phases             1 2222 4.589394 0.057553 28.979409  5.807404  64.252165  0.869037  0.883438  0.896940
    n_faulted_phases             2 4351 5.235442 0.030220 34.954763  9.242935  74.544838  0.856355  0.874282  0.884394
    n_faulted_phases             3 2214 5.260287 0.009713 35.902683  5.754667  89.373013  0.861789  0.883017  0.895212


## Interpretation (Reviewer #2.1)
With known line parameters, synchronized two-ended measurements, and a developed post-fault phasor, the classical double-ended method localizes essentially exactly (sub-1% for the large majority of cases) -- far below the DL error (~8% of line length at 50 ms). This CONFIRMS the reviewer: impedance methods are more accurate AND interpretable when their assumptions hold. The contribution of the DL approach is orthogonal: it achieves its accuracy **without any line parameters, without knowing the faulted segment a priori, and from a single forward pass on raw waveforms** -- and degrades gracefully, whereas the single-ended classical estimate is unstable. The two are complementary, not competing.
