# Distilling the Knowledge in a Neural Network 

- **Existing problem:**
  - For cumbersome models that learn to discriminate between a large number classes, the normal training objective is to maximize the average log probability of the correct answer, but a side-effect of the training is that the trained model assigns probabilities to all the incorrect answers and even when these probabilities are very small, some of them are much larger than others.

- **Main idea:**
  - Extract the distribution of training data from one *COMPLEX* model(Teacher Net) and "teach" the distribution to the *EASY* model(Student Net). The processed is called "distill".
  - <u>To raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets to be used for training the small model to match these soft target.</u>
  - e.g.
    - For the Teacher Network:
      - The probability of 3 is 0.998 which is ground-truth, and the probability of 5 and 7 are 0.004 and 0.005 respectively. 
      - $P('3') >> P('5')$ and $P('3') >> P('7')$
      - But $P('5')\approx P('7')$, which means Number '3' is somewhat like Number '2' or Number '7', this is the knowledge, image-label could not tell us but can be learned using an model(Teacher Network), is called 'dark knowledge'.
      - And now we want to 'teach' this 'dark knowledge' to another model(Student Network).
    - For the Student Network:
      - We wanna magnify this knowledge by 'distilling' it(use the formula below).
      - And learn this magnified 'dark knowledge' by making both this two networks output the same 'dark knowledge' with the same input.

- **Core Concept:**
  - $q_i=\frac{e^{z_i/T}}{\sum_je^{z_j/T}}=\frac{1}{\sum_je^{[(z_j-z_i)/T]}}$
    - Saying that $q_i$ is the "softmax" output layer that converts the logits, and $T$ is the temperature.
    - When $T=1$, $q_i$ is the "classical" calculation of "softmax".
    - As $T$ gets larger, here are three circumstances:
      - if $z_j > z_i$, $q_i$, which is large originally, will be smaller;
      - if $z_j < z_i$, $q_i$, which is small before, will be larger;
      - if $z_j = z_i$, $q_i$ just stays the same.
    - So the new distribution of **$q$** will be much balanced than before, solving the problem above.


