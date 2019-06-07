# Distilling the Knowledge in a Neural Network 

- **Existing problem:**
  - For cumbersome models that learn to discriminate between a large number classes, the normal training objective is to maximize the average log probability of the correct answer, but a side-effect of the training is that the trained model assigns probabilities to all the incorrect answers and even when these probabilities are very small, some of them are much larger than others.

- **Main idea:**
  - Extract the distribution of training data from one *COMPLEX* model(Teacher Net) and "teach" the distribution to the *EASY* model(Student Net). The processed is called "distill".
  - <u>To raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets to be used for training the small model to match these soft target.</u>

- **Core Concept:**

  - $q_i=\frac{e^{z_i/T}}{\sum_je^{z_j/T}}=\frac{1}{\sum_je^{[(z_j-z_i)/T]}}$
    - Saying that $q_i$ is the "softmax" output layer that converts the logits, and $T$ is the temperature.
    - When $T=1$, $q_i$ is the "classical" calculation of "softmax".
    - As $T$ gets larger, here are three circumstances:
      - if $z_j > z_i$, $q_i$, which is large originally, will be smaller;
      - if $z_j < z_i$, $q_i$, which is small before, will be larger;
      - if $z_j = z_i$, $q_i$ just stays the same.
    - So the new distribution of **$q$** will be much balanced than before, solving the problem above.


