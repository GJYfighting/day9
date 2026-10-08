# Day9 全部短实验记录

window includes first/inter-episode resets, grasp, randomization, inference and final recovery; excludes startup and network updates; parent stage wall values overlap children, never sum both

|实验|方法 / 配置|回合 / 成功|步数|完整窗口 s|秒 / 步|回合耗时 s|
|---|---|---:|---:|---:|---:|---|
|[correctness](20260923T020819_A_baseline_correctness_1790129299979688026/declaration.json)|A / baseline|1 / 1|4|79.964|19.991|77.04|
|[preparation](20260923T021103_B_baseline_preparation_1790129463791674096/declaration.json)|B / baseline|1 / 0|20|66.928|3.346|63.89|
|[preparation](20260923T021218_F_baseline_preparation_1790129538512981638/declaration.json)|F / baseline|1 / 1|4|88.051|22.013|85.14|
|[paired](20260923T021354_B_baseline_paired_1790129634200042985/declaration.json)|B / baseline|3 / 0|60|152.720|2.545|63.34, 44.06, 42.35|
|[closing_correctness](20260923T021722_A_baseline_closing_correctness_1790129842434426629/declaration.json)|A / baseline|1 / 0|4|68.983|17.246|53.90|
|[closing_handoff_correctness](20260923T021924_A_baseline_closing_handoff_correctness_1790129964845027950/declaration.json)|A / baseline|1 / 1|4|81.439|20.360|78.46|
|[paired](20260923T022219_F_baseline_paired_1790130139858827394/declaration.json)|F / baseline|3 / 3|12|201.318|16.777|79.78, 60.23, 58.29|
|[domain_regression](20260923T022549_F_baseline_domain_regression_1790130349445599232/declaration.json)|F / baseline|1 / 1|4|83.013|20.753|79.99|
|[paired](20260923T022928_B_open_servo_paired_1790130568020286283/declaration.json)|B / open_servo|3 / 0|60|206.670|3.445|77.06, 59.93, 58.81|
|[paired](20260923T023302_F_open_servo_paired_1790130782579660727/declaration.json)|F / open_servo|3 / 3|12|259.842|21.654|98.01, 76.14, 74.95|
|[paired](20260923T023738_B_reset_once_paired_1790131058417845803/declaration.json)|B / reset_once|3 / 0|60|146.444|2.441|60.71, 41.58, 41.21|
|[paired](20260923T024012_F_reset_once_paired_1790131212817114727/declaration.json)|F / reset_once|3 / 3|12|203.021|16.918|79.92, 59.28, 60.83|
|[paired](20260923T024509_B_no_resend_paired_1790131509421571671/declaration.json)|B / no_resend|3 / 0|60|157.566|2.626|66.03, 44.33, 44.24|
|[paired](20260923T024754_F_no_resend_paired_1790131674965460130/declaration.json)|F / no_resend|3 / 3|12|199.967|16.664|80.89, 58.52, 57.65|
|[paired](20260923T025251_B_combined_paired_1790131971339610062/declaration.json)|B / combined|3 / 0|60|180.932|3.016|65.87, 50.75, 53.55|
|[paired](20260923T025559_F_combined_paired_1790132159752470058/declaration.json)|F / combined|3 / 3|12|428.715|35.726|83.95, 148.98, 183.11|

旧闭合控制的 correctness / preparation 成功仅是原判据日志，不作为修复后物理回归通过。closing_correctness 保留了 NO_STALL 失败；closing_handoff_correctness 是后续修正成功。

每条声明含完整参数、代码哈希、模型来源、种子场景、回合/步数/墙钟/更新预算及恢复停止规则。原始事件、回合详情和失败均保留。

## 分项累计墙钟（父项与子项重叠，禁止逐列相加）

|方法 / 配置|复位总计|物理随机化|视觉|IK + 路径 + 步进 IK|动作执行|闭合|夹持稳定|抬升|保持|放回|松爪运动|松爪稳定|退离|回观察|最终恢复|推理|接近 / 接触 RTF|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B / baseline|64.110|3.611|2.693|10.541|83.745|0.000|0.000|0.000|0.000|0.000|16.820|1.609|0.000|11.144|2.938|0.064|0.860 / —|
|F / baseline|63.271|3.540|2.487|10.143|16.269|30.198|8.366|9.572|18.724|9.233|50.108|2.172|6.609|10.784|2.971|0.012|0.886 / 0.810|
|B / open_servo|110.841|3.507|2.678|11.200|83.079|0.000|0.000|0.000|0.000|0.000|66.364|1.717|0.000|11.075|10.836|0.060|0.867 / —|
|F / open_servo|111.569|3.387|2.627|10.059|16.287|20.007|7.658|9.520|19.228|9.619|113.043|2.347|6.594|10.803|10.708|0.013|0.885 / 0.816|
|B / reset_once|60.763|3.496|2.654|9.514|80.967|0.000|0.000|0.000|0.000|0.000|16.570|0.884|0.000|10.637|2.913|0.070|0.890 / —|
|F / reset_once|61.573|3.558|2.457|9.881|17.416|30.340|7.753|9.586|19.410|9.793|51.194|1.610|6.584|10.777|2.961|0.013|0.827 / 0.808|
|B / no_resend|66.365|3.646|2.710|11.111|86.228|0.000|0.000|0.000|0.000|0.000|17.940|1.626|0.000|11.016|2.927|0.074|0.836 / —|
|F / no_resend|64.591|3.497|2.407|10.362|16.125|28.986|7.827|9.505|18.567|9.153|50.015|2.217|6.585|10.767|2.880|0.013|0.894 / 0.835|
|B / combined|85.184|3.579|2.705|11.350|83.054|0.000|0.000|0.000|0.000|0.000|44.771|0.954|0.000|10.774|10.718|0.072|0.868 / —|
|F / combined|125.525|5.354|4.276|21.407|27.311|42.940|19.028|22.388|42.502|22.075|157.997|2.440|10.473|15.478|12.624|0.016|0.528 / 0.366|

启动时间、各项 exclusive 值及真实接触字段汇总另见 [measurements.json](measurements.json)。网络更新开销见 [update_probe.json](update_probe.json)，不与无学习窗口混加。
