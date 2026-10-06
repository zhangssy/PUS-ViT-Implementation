function results = calculate_metrics(gt, pred, eval_classes)
%CALCULATE_METRICS  Classification metrics for a PolSAR prediction map.
%   RESULTS = CALCULATE_METRICS(GT, PRED, EVAL_CLASSES)
%
%   GT and PRED are label maps of the same size. Only pixels whose ground-truth
%   label belongs to EVAL_CLASSES are scored. Background and every other
%   ground-truth label are ignored.
%
%   Class accuracy is the recall TP/(TP+FN). AA is the unweighted mean of those
%   recalls and is therefore the macro-average recall; it is not a separate
%   metric. MP is the unweighted mean of the per-class precisions TP/(TP+FP).
%   Macro F1 is the unweighted mean of the per-class F1 scores. OA is the
%   fraction of evaluated pixels that are correct. Kappa is Cohen's kappa, and
%   K100 is Kappa multiplied by 100.
%
%   A prediction whose label is outside EVAL_CLASSES is an error. It counts as
%   a false negative of its ground-truth class and is not assigned to another
%   evaluated class.
%
%   All ratios are stored on [0, 1], except K100, which is on the percentage
%   scale used in the manuscript.

if nargin < 3 || isempty(eval_classes)
    error('calculate_metrics:MissingClasses', ...
        'eval_classes must list the ground-truth labels to be scored.');
end

eval_classes = unique(eval_classes(:)', 'stable');
if any(eval_classes <= 0) || any(mod(eval_classes, 1) ~= 0)
    error('calculate_metrics:BadClasses', ...
        'eval_classes must contain positive integer labels.');
end

gt = squeeze(gt);
pred = squeeze(pred);
if ~isnumeric(gt) || ~isnumeric(pred) || ~ismatrix(gt) || ~ismatrix(pred)
    error('calculate_metrics:BadMaps', 'gt and pred must be 2-D numeric maps.');
end

transposed = false;
if ~isequal(size(gt), size(pred))
    if isequal(size(gt), fliplr(size(pred)))
        pred = pred.';
        transposed = true;
    else
        error('calculate_metrics:SizeMismatch', ...
            'Ground-truth size %s does not match prediction size %s.', ...
            mat2str(size(gt)), mat2str(size(pred)));
    end
end

valid = ismember(gt, eval_classes);
gt_eval = gt(valid);
pred_eval = pred(valid);
num_classes = numel(eval_classes);
n_evaluated = numel(gt_eval);
if n_evaluated == 0
    error('calculate_metrics:EmptyEvaluation', ...
        'No ground-truth pixel belongs to eval_classes.');
end

class_acc = zeros(num_classes, 1);
precision = zeros(num_classes, 1);
f1 = zeros(num_classes, 1);
support = zeros(num_classes, 1);
confusion = zeros(num_classes, num_classes);
n_predicted_outside = 0;

for i = 1:num_classes
    label = eval_classes(i);
    tp = sum((gt_eval == label) & (pred_eval == label));
    fn = sum((gt_eval == label) & (pred_eval ~= label));
    fp = sum((gt_eval ~= label) & (pred_eval == label));
    support(i) = tp + fn;

    if support(i) > 0
        class_acc(i) = tp / support(i);
    end
    if (tp + fp) > 0
        precision(i) = tp / (tp + fp);
    end
    if (precision(i) + class_acc(i)) > 0
        f1(i) = 2 * precision(i) * class_acc(i) / (precision(i) + class_acc(i));
    end
end

gt_index = zeros(n_evaluated, 1);
pred_index = zeros(n_evaluated, 1);
for i = 1:num_classes
    gt_index(gt_eval == eval_classes(i)) = i;
    pred_index(pred_eval == eval_classes(i)) = i;
end
inside = pred_index > 0;
n_predicted_outside = sum(~inside);
if any(inside)
    confusion = accumarray([gt_index(inside), pred_index(inside)], 1, ...
        [num_classes, num_classes]);
end

oa = sum(gt_eval == pred_eval) / n_evaluated;
aa = mean(class_acc);
mp = mean(precision);
macro_f1 = mean(f1);

pe = 0;
for i = 1:num_classes
    n_true = sum(gt_eval == eval_classes(i));
    n_pred = sum(pred_eval == eval_classes(i));
    pe = pe + double(n_true) * double(n_pred);
end
pe = pe / (double(n_evaluated) ^ 2);
if (1 - pe) == 0
    kappa = 1;
else
    kappa = (oa - pe) / (1 - pe);
end

results = struct();
results.eval_classes = eval_classes;
results.n_evaluated = n_evaluated;
results.support = support;
results.class_accuracy = class_acc;
results.precision = precision;
results.f1 = f1;
results.MP = mp;
results.macro_F1 = macro_f1;
results.OA = oa;
results.AA = aa;
results.Kappa = kappa;
results.K100 = 100 * kappa;
results.confusion = confusion;
results.n_predicted_outside = n_predicted_outside;
results.transposed = transposed;
end
