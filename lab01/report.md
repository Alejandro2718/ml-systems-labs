Lab 1: Environment and First System Measurements
1. Goal
I wanted to compare two machine learning techniques and see what each one offers at what computational cost. I compared accuracy, training time, prediction time, memory use, and model size. My goal was to understand when faster predictions or lower resource use matter more than a small difference in accuracy.
2. Method
I organized the folders using the lab manual's structure. I used Python 3.11.8 and saved the package versions in results/versions.txt. MLflow is still missing because its required version conflicts with the PyArrow version in the manual. This setup issue remains pending.
I used the Breast Cancer dataset: 569 samples with 30 features each. I split it into 398 training samples and 171 test samples using the 70/30 split with random_state=42. This kept similar class proportions in both groups. Both models used exactly the same data, without scaling.
I used Logistic Regression with max_iter=1000 and Random Forest with n_estimators=100, both with random_state=42. I also set the Python, NumPy, PyTorch, and CUDA seeds to 42.
I measured the following:
- Accuracy: I compared predictions with the test labels and saved accuracy to four decimal places in results/baseline_accuracy.csv.
- Training time: I ran one warm-up, then trained five fresh models of each type. I timed only training and reported the median, the middle value of the five times.
- Inference time: After one warm-up, I timed 100 separate predictions using the same single test sample and reported the median. Training and model loading were outside this timer.
- Model size: I saved each model with joblib.dump, without compression, and measured its size in bytes and KB. Each model has its own results subfolder containing model.joblib.
- Memory: I used psutil to sample process RSS during training and inference, requesting a sample every 1 ms. Each model and phase ran in a fresh process. Training covered one fit; inference covered 100 predictions after loading the model and warming up. I reported the highest observed RSS.
RSS includes Python, imported libraries, and the model. All measurements came from my Windows computer. I used decimal units: 1 KB = 1,000 bytes and 1 MB = 1,000,000 bytes.

3. Results
Measurement	Logistic Regression	Random Forest
Test accuracy	0.9415	0.9357
Correct predictions out of 171	161	160
Median training time (ms)	718.423	178.574
Median single-sample inference (ms)	0.062650	3.140700
Model size (bytes)	1,055	290,889
Model size (KB)	1.055	290.889
Observed training peak RSS (MB)	240.968	240.865
Observed inference peak RSS (MB)	240.951	242.176


Logistic Regression got one more prediction right. This difference is small and comes from one test split. It also reached the 1,000-iteration limit without converging. I recorded the warning and kept the settings required by the manual.
I compared the measurements with the manual's deployment budgets:
Environment	Memory budget	Maximum latency	Maximum model size
Cloud	>= 1 GB	100 ms	500 MB
Edge	256-1024 MB	50 ms	50 MB
Mobile	64-256 MB	20 ms	10 MB
TinyML	<= 256 KB	10 ms	100 KB


- Cloud: Both fit the numerical budgets: memory is below 1 GB, inference below 100 ms, and files below 500 MB.
- Edge: Both fit with 256 MB available: inference memory is below 256 MB, latency below 50 ms, and files below 50 MB.
- Mobile: Both fit at 256 MB, but not at 64 or 128 MB. They meet the 20 ms and 10 MB limits. Only about 15 MB and 14 MB remain, respectively, so other application components could exceed the memory budget.
- TinyML: Both fail the 256 KB memory limit in this Python setup. Logistic Regression meets the file size limit, but Random Forest also exceeds 100 KB. Both measured latencies are below 10 ms on my computer.
These comparisons do not prove performance on the target devices. Their speed can differ, and memory sampling can miss short peaks. The memory results describe this Python setup, not the minimum memory each model could use with another runtime.
4. Conclusions
1. I learned that training faster does not mean being more accurate. Random Forest trained about four times faster, but Logistic Regression got one more prediction right. Similar accuracy also does not mean similar resource use, even when both models use the same data.
2. For frequent predictions, I would choose Logistic Regression based on these results. What surprised me was that Random Forest trained faster, but took about 50 times longer for a prediction. Logistic Regression also had a much smaller model file.
3. I learned that file size and memory use are different. TinyML means running machine learning on very small devices with limited resources. My Logistic Regression file was only 1.055 KB, but the Python process used about 241 MB during inference. That is why this implementation failed the TinyML memory budget. A different runtime would need new measurements.
Sources: Lab Manual, Lab 1; Vijay Janapa Reddi, Introduction to Machine Learning Systems, assigned sections listed in the manual: Chapter 1, 1.4-1.6; Chapter 2, 2.2-2.5 and 2.8-2.9; Chapter 12, 12.4 and 12.7.