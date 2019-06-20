# Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images 

- **Main idea:**
  - All Neural Networks for classification do has a classifying result for every input image, unrecognizable image included.
  - Here comes the question: can we find unrecognizable images that DNNs will give out High confidence?
  - We initialize images with noises and using EA to select those images with high confidence. More details about EA is discussed in the next section.
- **Core Concept:**
  - Choosing a random organism(image) from the population, mutates it randomly, and replaces the current champion for any objective if the new individual has higher fitness(**less loss**) on that objective.
  - Direct encoding:
    - Each pixel value is initialized with uniform random noise with in the $[0,255]$ range;
    - Those numbers are independently mutated; first by determining which numbers are mutated, via a rate that starts at $0.1$ and drops by half every $1000$ generations; the numbers chosen to be mutated are then altered via the *polynomial mutation operator* with a fixed mutation strength of $15$.
  - Indirect encoding:
    - CPPN(compositional pattern-producing network)