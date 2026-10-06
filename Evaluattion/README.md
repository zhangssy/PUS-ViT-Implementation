# Prediction evaluation

This folder scores one PolSAR classification map against a ground-truth map. It implements the definitions used in the manuscript: class accuracy is recall, AA is the unweighted mean of those recalls, and MP is the unweighted mean of the per-class precisions.

## Input files

Both inputs are MAT files.

| File | Variable | Contents |
| --- | --- | --- |
| Ground truth | `cdata` | Integer label map. Background is `0`. |
| Prediction | `output` | Integer label map. Use the same label numbers as `cdata`. |

The two maps must have the same size. If one is the transpose of the other, the prediction is transposed before scoring.

## Metrics

Only pixels whose ground-truth label is listed in `eval_classes` are scored. Background and any other ground-truth label are ignored.

For class `c`:

- class accuracy = recall = `TP / (TP + FN)`
- precision = `TP / (TP + FP)`
- F1 = the harmonic mean of that precision and recall

`TP` is the number of evaluated pixels whose ground truth and prediction are both `c`. `FN` counts the remaining pixels whose ground truth is `c`. `FP` counts evaluated pixels predicted as `c` whose ground truth is a different evaluated class.

Aggregate scores:

- **MP**: unweighted mean of the per-class precisions
- **macro F1**: unweighted mean of the per-class F1 scores
- **OA**: fraction of evaluated pixels predicted correctly
- **AA**: unweighted mean of the class accuracies, which is the macro-average recall. AA is that quantity; a second recall average is not reported.
- **Kappa**: Cohen's kappa, using the reference and predicted class counts on the evaluated pixels
- **K100**: `100 * Kappa`

OA equals AA only when every evaluated class has the same number of pixels, or when the support-weighted and unweighted averages happen to coincide. A prediction label outside `eval_classes` is an error. It reduces recall and OA, and it is not added to another class as a false positive.

Ratios inside the saved struct are on `[0, 1]`. The console prints percentages. `K100` is already on the percentage scale.

## Run

From this folder in MATLAB, with no arguments, the function scores `result_acc.mat` against the Barnaul nine-class ground truth and writes `metrics.mat`:

```matlab
evaluate_prediction
```

To score another pair of maps:

```matlab
results = evaluate_prediction( ...
    'path/to/ground_truth.mat', ...
    'path/to/prediction.mat', ...
    1:9, ...
    'metrics.mat');
```

`1:9` is the set of land-cover labels to score. Replace it with the labels used by the map, such as `1:7`.

The fourth argument is optional. When it is given, every field is written to that single MAT file in the variable `results`.

## Saved fields

| Field | Meaning |
| --- | --- |
| `eval_classes` | Labels that were scored, in the reported order |
| `n_evaluated` | Number of scored pixels |
| `support` | Ground-truth count of each evaluated class |
| `class_accuracy` | Per-class recall |
| `precision` | Per-class precision |
| `f1` | Per-class F1 |
| `MP` | Macro-average precision |
| `macro_F1` | Macro-average F1 |
| `OA` | Overall accuracy |
| `AA` | Average accuracy (macro-average recall) |
| `Kappa` | Cohen's kappa |
| `K100` | Percentage kappa |
| `confusion` | Counts with rows as reference classes and columns as predicted classes, ordered as `eval_classes` |
| `n_predicted_outside` | Scored pixels whose prediction is not in `eval_classes` |
| `transposed` | True when the prediction map was transposed |
| `gt_path`, `pred_path` | Input paths |

`confusion(i, i)` is the true-positive count of `eval_classes(i)`. Pixels in `n_predicted_outside` are not placed in a confusion-matrix column.

## Files

- `evaluate_prediction.m` reads the two MAT files, prints the scores, and can save one results MAT file.
- `calculate_metrics.m` computes the scores from the two label maps.
