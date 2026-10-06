function results = evaluate_prediction(gt_path, pred_path, eval_classes, save_path)
%EVALUATE_PREDICTION  Score one prediction map and optionally save one MAT file.
%   RESULTS = EVALUATE_PREDICTION(GT_PATH, PRED_PATH, EVAL_CLASSES)
%   RESULTS = EVALUATE_PREDICTION(GT_PATH, PRED_PATH, EVAL_CLASSES, SAVE_PATH)
%
%   GT_PATH is a MAT file containing the ground-truth map in variable cdata.
%   PRED_PATH is a MAT file containing the prediction map in variable output.
%   EVAL_CLASSES lists the land-cover labels to score, for example 1:7 or 1:9.
%   Labels outside this list, including background 0, are not evaluated.
%
%   SAVE_PATH, when supplied, is the single MAT file that receives RESULTS.
%   The saved variable name is results.
%
%   Calling EVALUATE_PREDICTION with no arguments scores the Barnaul example
%   packaged with this folder and writes metrics.mat beside this file.

if nargin == 0
    gt_path = '';
    pred_path = fullfile(fileparts(mfilename('fullpath')), 'result_acc.mat');
    eval_classes = 1:9;
    save_path = fullfile(fileparts(mfilename('fullpath')), 'metrics.mat');
elseif nargin < 3
    error('evaluate_prediction:NotEnoughInputs', ...
        'Usage: evaluate_prediction(gt_path, pred_path, eval_classes, save_path).');
elseif nargin < 4
    save_path = '';
end

gt = load_map(gt_path, 'cdata', 'ground truth');
pred = load_map(pred_path, 'output', 'prediction');
results = calculate_metrics(gt, pred, eval_classes);
results.gt_path = char(gt_path);
results.pred_path = char(pred_path);

print_results(results);

if ~isempty(save_path)
    save(save_path, 'results');
    fprintf('Saved results to %s\n', save_path);
end
end

function map = load_map(mat_path, variable_name, role_name)
if ~exist(mat_path, 'file')
    error('evaluate_prediction:MissingFile', ...
        'Cannot find the %s file: %s', role_name, mat_path);
end
data = load(mat_path);
if ~isfield(data, variable_name)
    error('evaluate_prediction:MissingVariable', ...
        ['The %s file must contain variable "%s". Found: %s'], ...
        role_name, variable_name, strjoin(fieldnames(data), ', '));
end
map = data.(variable_name);
end

function print_results(results)
fprintf('\nEvaluated classes: %s\n', mat2str(results.eval_classes));
fprintf('Evaluated pixels:  %d\n', results.n_evaluated);
if results.transposed
    fprintf('The prediction map was transposed to match the ground truth.\n');
end
if results.n_predicted_outside > 0
    fprintf(['Pixels predicted outside the evaluated classes: %d ', ...
        '(counted as errors, omitted from the confusion-matrix columns).\n'], ...
        results.n_predicted_outside);
end

fprintf('\nPer-class metrics (%%)\n');
fprintf('%-8s %10s %10s %10s %10s\n', 'Class', 'Accuracy', 'Precision', 'F1', 'Support');
for i = 1:numel(results.eval_classes)
    fprintf('%-8d %10.2f %10.2f %10.2f %10d\n', ...
        results.eval_classes(i), ...
        100 * results.class_accuracy(i), ...
        100 * results.precision(i), ...
        100 * results.f1(i), ...
        results.support(i));
end

fprintf('\nAggregate metrics\n');
fprintf('MP (macro-average precision) = %.2f%%\n', 100 * results.MP);
fprintf('Macro-average F1             = %.2f%%\n', 100 * results.macro_F1);
fprintf('OA                           = %.2f%%\n', 100 * results.OA);
fprintf('AA (macro-average recall)    = %.2f%%\n', 100 * results.AA);
fprintf('Kappa                        = %.4f\n', results.Kappa);
fprintf('K100                         = %.2f\n', results.K100);
end
