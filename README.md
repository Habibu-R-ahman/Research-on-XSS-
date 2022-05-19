## Contents

- [An implementation of real-time detection of cross-site scripting attacks on cloud-based web applications using deep learning](#intro1)
- [DeepXSS: Cross Site Scripting Detection Based on Deep Learning](#intro2) 
- [Statically Identifying XSS using Deep Learning](#intro3)
- [A Distributed Deep Learning System for Web Attack Detection on Edge Devices](#intro4)
- [CODDLE: Code-Injection Detection With Deep Learning](#intro5)
- [XSS Detection Technology Based on LSTM-Attention](#intro6)
- [New deep learning method to detect code injection attacks on hybrid applications](#intro7)
- [Adversarial Examples Detection for XSS Attacks Based on Generative Adversarial Networks](#intro8)
- [Detecting web attacks with end-to-end deep learning](#intro9)
- [Locate-Then-Detect: Real-time Web Attack Detection via Attention-based Deep Neural Networks](#intro10)
- [A hybrid of CNN and LSTM methods for securing web application against cross-site scripting attack](#intro11)
- [MLPXSS: An Integrated XSS-Based Attack Detection Scheme in Web Applications Using Multilayer Perceptron Technique](#intro12)
- [MECHINE LEARNING](#intro13)
- [A Detailed Survey on Recent XSS Web-Attacks Machine Learning Detection Techniques](#intro14)
- [Towards a Lightweight, Hybrid Approach for Detecting DOM XSS Vulnerabilities with Machine Learning](#intro15)
- [A Comparison of Machine Learning Algorithms for Detecting XSS Attacks](#intro16)
- [XSSClassifier: An Efficient XSS Attack Detection Approach Based on Machine Learning Classifier on SNSs](#intro17)
- [XSS Attack Detection With Machine Learning and n-Gram Methods](#intro18)
- [An ensemble learning approach for XSS attack detection with domain knowledge and threat intelligence](#intro19)
- [Prediction of Cross-Site Scripting Attack Using Machine Learning Algorithms](#intro20)
- [Detecting Cross-Site Scripting Attacks Using Machine Learning](#intro21)
- [Machine Learning Based Cross-Site Scripting Detection in Online Social Network](#intro22)
- [Detection of XSS Attacks in Web Applications: A Machine Learning Approach](#intro23)
- [Cross-site Scripting Attack Detection Using Machine Learning with Hybrid Features](#intro24)
- [Detecting Blind Cross-Site Scripting Attacks Using Machine Learning](#intro25)
- [XGBXSS: An Extreme Gradient Boosting Detection Framework for Cross-Site Scripting Attacks Based on Hybrid Feature Selection Approach and Parameters Optimization](#intro26)
- [RLXSS: Optimizing XSS Detection Model to Defend Against Adversarial Attacks Based on Reinforcement Learning](#intro27)
- [EXTRA](#intro28)
- [A Survey of Exploitation and Detection Methods of XSS Vulnerabilities](#intro29)
- [XSSDS: Server-Side Detection of Cross-Site Scripting Attacks](#intro30)
- [Noxes: A client-side solution for mitigating cross-site scripting attacks](#intro31)


## DEEP LEARNING METHODS

<a name="intro1"></a>
### 1. An implementation of real-time detection of cross-site scripting attacks on cloud-based web applications using deep learning
- DOI: https://doi.org/10.11591/eei.v10i5.3168
- Cited by: 1
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/An_implementation_of_real-time_detection_of_cross-.pdf)
- <details>
  <summary>Summary</summary>
  <p>This project work utilized five phases cross-site scripting payloads and Benign user inputs extraction, feature engineering, generation of datasets, deep learning modeling, and classification filter for Malicious cross-site scripting queries. A web application was then developed with the deep learning model embedded on the backend and hosted on the cloud. In this work, a model was developed to detect cross-site scripting attacks using multi-layer perceptron deep learning model, after a comparative analysis of its performance in contrast to three other deep learning models deep belief network, ensemble, and long short-term memory. A multi-layer perceptron based performance evaluation of the proposed model obtained an accuracy of 99.47%, which shows a high level of accuracy in detecting cross-site scripting attacks.</p>
  </details>  

<a name="intro2"></a>
### 2. DeepXSS: Cross Site Scripting Detection Based on Deep Learning
- DOI: https://doi.org/10.1145/3194452.3194469
- Cited By: 31
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/DeepXSS.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this paper, we present a novel approach to detect XSS attacks based on deep learning (called DeepXSS). First of all, we used word2vec to extract the feature of XSS payloads which captures word order information and map each payload to a feature vector. And then, we trained and tested the detection model using Long Short Term Memory (LSTM) recurrent neural networks. Experimental results show that the proposed XSS detection model based on deep learning achieves a precision rate of 99.5% and a recall rate of 97.9% in real dataset, which means that the novel approach can effectively identify XSS attacks</p>
  </details> 

<a name="intro3"></a>
### 3. Statically Identifying XSS using Deep Learning
- DOI: https://doi.org/10.1016/j.scico.2022.102810
- Cited By: 1
- PDF: [Download]()
- <details>
  <summary>Summary</summary>
  <p>This work explores static approaches to detect XSS vulnerabilities using neural networks. We compare two different code representations based on Natural Language Processing (NLP) and Programming Language Processing (PLP) and experiment with models based on different neural network architectures for static analysis detection in PHP and Node.js. We train and evaluate the models using synthetic databases. Using the generated PHP and Node.js databases, we compare our results with three well-known static analyzers for PHP code, ProgPilot, Pixy, RIPS, and a known scanner for Node.js, AppScan static mode. Our analyzers using neural networks overperform the results of existing tools in all cases.</p>
  </details> 

<a name="intro4"></a>
### 4. A Distributed Deep Learning System for Web Attack Detection on Edge Devices
- DOI: https://doi.org/10.1109/TII.2019.2938778	
- Cited By: 131
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/A%20Distributed%20Deep%20Learning%20System%20for%20Web.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this paper, based on distributed deep learning, we propose a web attack detection system that takes advantage of analyzing URLs. The system is designed to detect web attacks and is deployed on edge devices. The cloud handles the above challenges in the paradigm of the Edge of Things (EoT). Multiple concurrent deep models are used to enhance the stability of the system and the convenience in updating. We implemented experiments on the system with two concurrent deep models and compared the system with existing systems by using several datasets. The experimental results with 99.410% in accuracy, 98.91% in TPR and 99.55% in DRN demonstrate the system is competitive in detecting web attacks.</p>
  </details> 


<a name="intro5"></a>
### 5. CODDLE: Code-Injection Detection With Deep Learning
- DOI: https://doi.org/10.1109/ACCESS.2019.2939870
- Cited by: 16
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/CODDLE_Code-Injection_Detection_With_Deep_Learning.pdf)
- <details>
  <summary>Summary</summary>
  <p>This paper proposes CODDLE, a deep learning-based intrusion detection systems against web-based code injection attacks. CODDLE’s main novelty consists in adopting a Convolutional Deep Neural Network and in improving its effectiveness via a tailored pre-processing stage which encodes SQL/XSS-related symbols into type/value pairs. Numerical experiments performed on real-world datasets for both SQL and XSS attacks show that, with an identical training and with a same neural network shape, CODDLE’s type/value encoding improves the detection rate from a baseline of about 75% up to 95% accuracy, 99% precision, and a 92% recall value</p>
  </details> 

<a name="intro6"></a>
### 6. XSS Detection Technology Based on LSTM-Attention
- DOI: https://doi.org/10.1109/CRC51253.2020.9253484
- Cited By: 2
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/XSS%20Detection%20Technology%20Based%20on%20LSTM-Attention.pdf)
- <details>
  <summary>Summary</summary>
  <p>we present a novel approach to detect XSS attacks based on the attention mechanism of Long Short-Term Memory (LSTM) recurrent neural network. First of all, the data need to be preprocessed, we used decoding technology to restore the XSS codes to the unencoded state for improving the readability of the code, then we used word2vec to extract XSS payload features and map them to feature vectors. And then, we improved the LSTM model by adding attention mechanism, the LSTM-Attention detection model was designed to train and test the data. We used the ability of LSTM model to extract context-related features for deep learning, the added attention mechanism made the model extract more effective features. Finally, we used the classifier to classify the abstract features. Experimental results show that the proposed XSS detection model based on LSTM-Attention achieves a precision rate of 99.3% and a recall rate of 98.2% in the actually collected dataset. Compared with traditional machine learning methods and other deep learning methods, this method can more effectively identify XSS attacks.</p>
  </details> 

<a name="intro7"></a>
### 7. New deep learning method to detect code injection attacks on hybrid applications	
- DOI: https://doi.org/10.1016/j.jss.2017.11.001
- Cited By: 21
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/New%20deep%20learning%20method.pdf)
- <details>
  <summary>Summary</summary>
  <p>We present a novel approach to detect XSS attacks based on the attention mechanism of Long Short-Term Memory (LSTM) recurrent neural network. First of all, the data need to be preprocessed, we used decoding technology to restore the XSS codes to the unencoded state for improving the readability of the code, then we used word2vec to extract XSS payload features and map them to feature vectors. And then, we improved the LSTM model by adding attention mechanism, the LSTM-Attention detection model was designed to train and test the data. We used the ability of LSTM model to extract context-related features for deep learning, the added attention mechanism made the model extract more effective features. Finally, we used the classifier to classify the abstract features. Experimental results show that the proposed XSS detection model based on LSTM-Attention achieves a precision rate of 99.3% and a recall rate of 98.2% in the actually collected dataset. Compared with traditional machine learning methods and other deep learning methods, this method can more effectively identify XSS attacks.</p>
  </details> 


<a name="intro8"></a>
### 8. Adversarial Examples Detection for XSS Attacks Based on Generative Adversarial Networks
- DOI: https://doi.org/10.1109/ACCESS.2020.2965184
- Cited By: 18
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/Adversarial_Examples_Detection_for_XSS_Attacks_Based_on_Generative_Adversarial_Networks.pdf)
- <details>
  <summary>Summary</summary>
  <p>Models based on deep learning are prone to misjudging the results when faced with adversarial examples. In this paper, we propose an MCTS-T algorithm for generating adversarial examples of cross-site scripting (XSS) attacks based on Monte Carlo tree search (MCTS) algorithm. The MCTS algorithm enables the generation model to provide a reward value that reflects the probability of generative examples bypassing the detector. To guarantee the antagonism and feasibility of the generative adversarial examples, the bypassing rules are restricted. The experimental results indicate that the missed detection rate of adversarial examples is significantly improved after the MCTS-T generation algorithm. Additionally, we construct a generative adversarial network (GAN) to optimize the detector and improve the detection rate when dealing with adversarial examples. After several epochs of adversarial training, the accuracy of detecting adversarial examples is significantly improved.</p>
  </details> 

<a name="intro9"></a>
### 9. Detecting web attacks with end-to-end deep learning
- DOI: https://doi.org/10.1186/s13174-019-0115-x
- Cited By: 34
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/Detecting_web_attacks_with_end-to-end_deep_learnin.pdf)
- <details>
  <summary>Summary</summary>
  <p>This paper provides three contributions to the study of autonomic intrusion detection systems. First, we evaluate the feasibility of an unsupervised/semi-supervised approach for web attack detection based on the Robust Software Modeling Tool (RSMT), which autonomically monitors and characterizes the runtime behavior of web applications. Second, we describe how RSMT trains a stacked denoising autoencoder to encode and reconstruct the call graph for end-to-end deep learning, where a low-dimensional representation of the raw features with unlabeled request data is used to recognize anomalies by computing the reconstruction error of the request data. Third, we analyze the results of empirically testing RSMT on both synthetic datasets and production applications with intentional vulnerabilities. Our results show that the proposed approach can efficiently and accurately detect attacks, including SQL injection, cross-site scripting, and deserialization, with minimal domain knowledge and little labeled training data</p>
  </details> 

<a name="intro10"></a>
### 10. Locate-Then-Detect: Real-time Web Attack Detection via Attention-based Deep Neural Networks
- DOI: https://doi.org/10.24963/ijcai.2019/656
- Cited By: 16
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/Locate-Then-Detect:.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this study, we propose a novel Locate-Then-Detect (LTD) system that can precisely detect Web threats in real-time by using attention-based deep neural networks. Firstly, an efficient Payload Locating Network (PLN) is employed to propose most suspicious regions from large URL requests/posts. Then a Payload Classification Network (PCN) is adopted to accurately classify malicious regions from suspicious candidates. In this way, PCN can focus more on learning malicious segments and highly increase detection accuracy. The noise induced by irrelevant background strings can be largely eliminated. Besides, LTD can greatly reduce computational costs (82.6% less) by ignoring large irrelevant URL content</p>
  </details> 

<a name="intro11"></a>
### 11. A hybrid of CNN and LSTM methods for securing web application against cross-site scripting attack
- DOI: https://doi.org/10.11591/ijeecs.v21.i2.pp1022-1029
- Cited By: 6
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/A_hybrid_of_CNN_and_LSTM_methods_for_securing_web_.pdf)
- <details>
  <summary>Summary</summary>
  <p>In first part of this paper, a method for detecting XSS attack was proposed by combining convolutional neural network (CNN) with long short term memories ( LSTM), Initially, pre-processing was applied to XSS Data Set by decoding, generalization and tokanization, and then word2vec was applied to convert words into word vectors in XSS payloads. And then we use the combination CNN with LSTM to train and test word vectors to produce a model that can be used in a web application. Based on the obtaned results, it is observed that the proposed model achevied an excellent result with accuracy of 99.4%.</p>
  </details> 

<a name="intro12"></a>
### 12. MLPXSS: An Integrated XSS-Based Attack Detection Scheme in Web Applications Using Multilayer Perceptron Technique
- DOI: https://doi.org/10.1109/ACCESS.2019.2927417
- Cited By: 31
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/MLPXSS_An_Integrated_XSS-Based_Attack_Detection_Sc.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this research, a robust artificial neural network-based multilayer perceptron (MLP) scheme integrated with the dynamic feature extractor is proposed for XSS attack detection. The detection scheme adopts a large real-world dataset, the dynamic features extraction mechanism, and MLP model, which successfully surpassed several tests on an employed unique dataset under careful experimentation, and achieved promising and state-of-the-art results with accuracy, detection probabilities, false positive rate, and AUC-ROC scores of 99.32%, 98.35%, 0.3%, and 90.02% respectively. Therefore, it has the potentials to be applied for XSS based attack detection in either the client-side or the server-side.</p>
  </details> 
  
<a name="intro13"></a>
## MACHINE LEARNING METHODS

<a name="intro14"></a>
### 1. A Detailed Survey on Recent XSS Web-Attacks Machine Learning Detection Techniques
- DOI: https://doi.org/10.1109/GCAT52182.2021.9587569
- Cited By: 1
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/ADetailedSurveyonRecentXSSWeb-AttacksMachineLearningDetectionTechniquesPDF.pdf)
- <details>
  <summary>Summary</summary>
  <p>There are ample of methodologies being applied in the detection of XSS attacks using supervised learning, unsupervised learning, reinforcement learning, deep learning and metaheuristic algorithms. We present a survey of the recent approaches being applied by the numerous researchers in their proposed models. Following indexed journals were used for research papers’ collection in order to carry out a survey: Elsevier, Springer, IEEE explore, Hindawi, google scholar, and Web of Science. Moreover, in this paper, we introduce a classification chart of several machine learning algorithms that can be applied to the web-attack detection model.</p>
  </details>  

<a name="intro15"></a>
### 2. Towards a Lightweight, Hybrid Approach for Detecting DOM XSS Vulnerabilities with Machine Learning
- DOI: https://doi.org/10.1145/3442381.3450062
- Cited By: 5
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/Towards%20a%20Lightweight.pdf)
- <details>
  <summary>Summary</summary>
  <p>Through a large-scale web crawl, we collect over 18 billion JavaScript functions and use taint tracking to label over 180,000 functions as potentially vulnerable. With this data, we train a deep neural network (DNN) to analyze a JavaScript function and predict if it is vulnerable to DOM XSS. We experiment with a range of hyperparameters and present a low-latency, high-recall classifier that could serve as a pre-filter to taint tracking, reducing the cost of stand-alone taint tracking by 3.43 × while detecting 94.5% of unique vulnerabilities. We argue that this combination of a DNN and taint tracking is efficient enough for a range of use cases for which taint tracking by itself is not, including in-browser run-time DOM XSS detection and analyzing large codebases.</p>
  </details> 

<a name="intro16"></a>
### 3. A Comparison of Machine Learning Algorithms for Detecting XSS Attacks
- DOI: https://doi.org/10.1007/978-3-030-24268-8_20
- Cited By: 7
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/A%20Comparison%20of%20Machine%20Learning%20Algorithms%20for%20Detecting%20XSS%20Attacks.pdf)
- <details>
  <summary>Summary</summary>
  <p>this paper summarizes the method of XSS recognition based on the machine learning algorithm, classifies different machine learning algorithms according to the recognition strategy, analyzes their advantages and disadvantages, and finally looks forward to the development trend of XSS defense research, hoping to play a reference role for the following researchers.</p>
  </details> 

<a name="intro17"></a>
### 4. XSSClassifier: An Efficient XSS Attack Detection Approach Based on Machine Learning Classifier on SNSs	
- DOI: https://doi.org/10.3745/JIPS.03.0079
- Cited By: 47
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/XSSClassifier_%20An%20Efficient%20XSS%20Attack%20Detection%20Approach%20Based%20on%20Machine%20Learning%20Classifier%20on%20SNSs.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this paper, we propose a machine learningbased approach to detecting XSS attack on SNSs such as Twitter, MySpace, and Facebook. In our approach, the detection of XSS attack is performed based on three features: URLs, webpage, and SNSs. A dataset is prepared by collecting 1,000 SNSs webpages and extracting the features from these webpages. Ten different machine learning classifiers are used on a prepared dataset to classify webpages into two categories: XSS or non-XSS. To validate the efficiency of the proposed approach, we evaluated and compared it with other existing approaches. The evaluation results show that our approach attains better performance in the SNS environment, recording the highest accuracy of 0.972 and lowest false positive rate of 0.87</p>
  </details> 

<a name="intro18"></a>
### 5. XSS Attack Detection With Machine Learning and n-Gram Methods
- DOI: https://doi.org/10.1109/ICIMTech50083.2020.9210946
- Cited By: 4
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/XSS%20Attack%20Detection%20With%20Machine%20Learning%20and%20n-Gram%20Methods.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this study, the authors conducted a study by evaluating several machine learning methods, namely Support Vector Machine (SVM), K-Nearest Neighbour (KNN), and Naïve Bayes (NB). The machine learning algorithm is then equipped with the n-gram method to each script feature to improve the detection performance of XSS attacks. The simulation results show that the SVM and n-gram method achieves the highest accuracy with 98%.</p>
  </details> 

<a name="intro19"></a>
### 6. An ensemble learning approach for XSS attack detection with domain knowledge and threat intelligence
- DOI: https://doi.org/10.1016/j.cose.2018.12.016
- Cited by: 35
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/An%20ensemble%20learning%20approach%20for%20XSS%20attack%20detection%20with%20domain%20knowledge%20and%20threat%20intelligence.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this paper, the XSS attack detection method is proposed based on an ensemble learning approach which utilizes a set of Bayesian networks, and each Bayesian network is built with both domain knowledge and threat intelligence. Besides, an analysis method is proposed to further explain the results, which sorts nodes in the Bayesian network according to their influences on the output node. The results are explainable to the end users. To validate the proposed method, experiments are performed on a real-world dataset about the XSS attack. The results show the priority of the proposed method, especially when the number of attacks inc# Welcome to StackEdit!</p>
  </details> 


<a name="intro20"></a>
### 7. Prediction of Cross-Site Scripting Attack Using Machine Learning Algorithms
- DOI: https://doi.org/10.1145/2660859.2660969
- Cited By: 20
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/Prediction_of_XSS.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this paper, we present the experimental results obtained using three machine learning algorithms (Naïve Bayes, Support Vector Machine and J48 Decision Tree) for the prediction of Cross-site scripting attack. This is done using the features based on normal and malicious URLs and JavaScript. J48 gave better results than Naïve Bayes and Support Vector Machine based on the features extracted from URL and Java Script code. All the algorithms gave comparatively better results with discretized attributes but noticeable difference in performance was seen only in the case of SVM.</p>
  </details> 

<a name="intro21"></a>
### 8. Detecting Cross-Site Scripting Attacks Using Machine Learning
- DOI: https://doi.org/10.1007/978-3-319-74690-6_20
- Cited By: 13
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/DetectingCross-SiteScriptingAttacksUsingMachineLearning.pdf)
- <details>
  <summary>Summary</summary>
  <p>This paper investigates using SVM, k-NN and Random Forests to detect and limit these attacks, whether known or unknown, by building classifiers for JavaScript code. It demonstrated that using an interesting feature set combining language syntax and behavioural features results in classifiers that give high accuracy and precision on large real world data sets without restricting attention only to obfuscation.</p>
  </details> 

<a name="intro22"></a>
### 9. Machine Learning Based Cross-Site Scripting Detection in Online Social Network
- DOI: https://doi.org/10.1109/HPCC.2014.137
- Cited By: 16
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/Machine%20Learning%20Based%20Cross-Site%20Scripting%20Detection%20in%20Online%20Social%20Network.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this paper, we present a novel approach using machine learning to do XSS detection in OSN. Firstly, we leverage a new method to capture identified features from web pages and then establish classification models which can be used in XSS detection. Secondly, we propose a novel method to simulate XSS worm spreading and build our webpage database. Finally, we set up experiments to verify the classification models using our test database. Our experiment results demonstrate that our approach is an effective countermeasure to detect the XSS attack.</p>
  </details> 

<a name="intro23"></a>
### 10. Detection of XSS Attacks in Web Applications: A Machine Learning Approach
- DOI: https://doi.org/10.21276/ijircst.2021.9.1.1
- Cited By: 1
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/raw/main/PDF/1-detection-of-xss-attacks-in-web-applications-a-machine-learning-approach.pdf)
- <details>
  <summary>Summary</summary>
  <p>As such, rule-based and signature-based web application firewalls are not effective against detecting XSS attacks for payloads designed to bypass web application firewalls. This paper aims to use machine learning to detect XSS attacks using various ML (machine learning) algorithms and to compare the performance of the algorithms in detecting XSS attacks in web applications and websites.</p>
  </details> 

<a name="intro24"></a>
### 11. Cross-site Scripting Attack Detection Using Machine Learning with Hybrid Features	
- DOI: https://doi.org/10.20895/infotel.v13i1.606
- Cited By: 1
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/Cross-site%20Scripting%20Attack%20Detection%20Using%20Machine%20Learning%20with%20Hybrid%20Features.pdf)
- <details>
  <summary>Summary</summary>
  <p>This study aims to measure the classification accuracy of XSS attacks by using a combination of two methods of determining feature characteristics, namely using linguistic computation and feature selection. XSS attacks have a certain pattern in their character arrangement, this can be studied by learners using n-gram modeling, but in certain cases XSS characteristics can contain a certain meta and synthetic this can be learned using feature selection modeling. From the results of this research, hybrid feature modeling gives good accuracy with an accuracy value of 99.87%, it is better than previous studies which the average is still below 99%, this study also tries to analyze the false positive rate considering that the false positive rate in attack detection is very influential for the convenience of the information security team, with the modeling proposed, the false positive rate is very small, namely 0.039%</p>
  </details> 

<a name="intro25"></a>
### 12. Detecting Blind Cross-Site Scripting Attacks Using Machine Learning
- DOI: https://doi.org/10.1145/3297067.3297096
- Cited by: 5
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/Detecting%20Blind%20Cross-Site%20Scripting%20Attacks%20Using%20Machine%20Learning.pdf)
- <details>
  <summary>Summary</summary>
  <p>Most of the XSS detection techniques used to detect the XSS vulnerabilities are inadequate to detect blind XSS attacks. In this research, we present machine learning based approach to detect blind XSS attacks. Testing results help to identify malicious payloads that are likely to get stored in databases through web applications.</p>
  </details> 

<a name="intro26"></a>
### 13. XGBXSS: An Extreme Gradient Boosting Detection Framework for Cross-Site Scripting Attacks Based on Hybrid Feature Selection Approach and Parameters Optimization
- DOI: https://doi.org/10.1016/j.jisa.2021.102813
- Cited By: 3
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/XGBXSS-preprint.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this study, we proposed XGBXSS, a novel web-based XSS attack detection framework based on an ensemble-learning technique using the Extreme Gradient Boosting algorithm (XGboost) with extreme parameters optimization approach. An enhanced feature extraction method is presented to extract the most useful features from the developed dataset. Furthermore, a novel hybrid approach for features selection is proposed, comprising information gain (IG) fusing with sequential backward selection (SBS) to select an optimal subset reducing the computational costs and maintaining the high-performance of detector' simultaneously. The proposed framework has successfully exceeded several tests on the holdout testing dataset and achieved avant-garde results with accuracy, precision, detection probabilities, F-score, false-positive rate, false-negative rate, and AUC-ROC scores of 99.59%, 99.53 %, 99.01%, 99.27%, 0.18%, 0.98%, and 99.41%, respectively</p>
  </details> 

<a name="intro27"></a>
### 14. RLXSS: Optimizing XSS Detection Model to Defend Against Adversarial Attacks Based on Reinforcement Learning
- DOI: https://doi.org/10.3390/fi11080177
- Cited By: 7
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/RLXSS_Optimizing_XSS_Detection_Model_to_Defend_Aga.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this paper, we present a method based on reinforcement learning (called RLXSS), which aims to optimize the XSS detection model to defend against adversarial attacks. First, the adversarial samples of the detection model are mined by the adversarial attack model based on reinforcement learning. Secondly, the detection model and the adversarial model are alternately trained. After each round, the newly-excavated adversarial samples are marked as a malicious sample and are used to retrain the detection model. Experimental results show that the proposed RLXSS model can successfully mine adversarial samples that escape black-box and white-box detection and retain aggressive features. What is more, by alternately training the detection model and the confrontation attack model, the escape rate of the detection model is continuously reduced, which indicates that the model can improve the ability of the detection model to defend against attacks.</p>
  </details> 

<a name="intro28"></a>
## Out of Context But Worth Reading

<a name="intro29"></a>
### 1. A Survey of Exploitation and Detection Methods of XSS Vulnerabilities
- DOI: https://doi.org/10.1109/ACCESS.2019.2960449
- Cited by: 17
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/A_Survey_of_Exploitation_and_Detection_Methods_of_.pdf)
- <details>
  <summary>Summary</summary>
  <p>This paper discusses classification of XSS, and designs a demo website to demonstrate attack processes of common XSS exploitation scenarios. The paper also compares and analyzes recent research results on XSS detection, divides them into three categories according to different mechanisms. The three categories are static analysis methods, dynamic analysis methods and hybrid analysis methods. The paper classifies 30 detection methods into above three categories, makes overall comparative analysis among them, lists their strengths and weaknesses and detected XSS vulnerability types. In the end, the paper explores some ways to prevent XSS vulnerabilities from being exploited.</p>
  </details> 


<a name="intro30"></a>
### 2. XSSDS: Server-Side Detection of Cross-Site Scripting Attacks
- DOI: https://doi.org/10.1109/ACSAC.2008.36
- Cited By: 76
- PDF: [Download](https://github.com/Habibu-R-ahman/Reseach-on-XSS-/blob/main/PDF/XSSDS_Server-Side_Detection_of_Cross-Site_Scriptin.pdf)
- <details>
  <summary>Summary</summary>
  <p>In this paper, we propose a passive detection system to identify successful XSS attacks. Based on a prototypical implementation, we examine our approach's accuracy and verify its detection capabilities. We compiled a data-set of 500.000 individual HTTP request/response-pairs from 95 popular web applications for this, in combination with both real word and manually crafted XSS-exploits; our detection approach results in a total of zero false negatives for all tests, while maintaining an excellent false positive rate for more than 80% of the examined Web applications.</p>
  </details> 

<a name="intro31"></a>
### 3. Noxes: A client-side solution for mitigating cross-site scripting attacks
- DOI: https://doi.org/10.1145/1141277.1141357
- Cited By: 291
- PDF: [Download]()
- <details>
  <summary>Summary</summary>
  <p>This paper presents Noxes, which is, to the best of our knowledge, the first client-side solution to miti- gate cross-site scripting attacks. Noxes acts as a web proxy and uses both manual and automatically generated rules to mitigate possible cross-site scripting attempts. Noxes effec- tively protects against information leakage from the user's environment while requiring minimal user interaction and customization effort.</p>
  </details> 


