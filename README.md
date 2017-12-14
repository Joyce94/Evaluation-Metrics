# Evaluation-Metrics
这是一份命名实体识别任务的评测部分的脚本。

用了三种评价指标方法：Exact Match、Binary Overlap、Proportional Overlap，使用了十折交叉验证，计算F值，计算结果保存在了output_my.txt中。

注意：

在计算Binary Overlap和Proportional Overlap时，很容易出错，在计算Precision和Recall的分子时，应该分别遍历正确序列和预测序列，得到了分子，即预测正确的个数，应该是不同的，而我们很容易误认为是相同。

若是按照这种做法，则十折交叉验证之后会得到output_error.txt中的结果。
