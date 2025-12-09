# Activation Function Analysis Report

**Name:** _______________  
**Date:** _______________

---

## 1. Output Range Comparison

| Activation | Range | Best For |
|------------|-------|----------|
| Step       |       |          |
| Sigmoid    |       |          |
| Tanh       |       |          |
| ReLU       |       |          |
| Leaky ReLU |       |          |

---

## 2. Gradient Behavior

### Sigmoid

- At z = 0: gradient = ___
- At z = 5: gradient = ___
- At z = -5: gradient = ___
- Problem identified: 

### Tanh

- At z = 0: gradient = ___
- At z = 5: gradient = ___
- Problem identified:

### ReLU

- At z > 0: gradient = ___
- At z < 0: gradient = ___
- Problem identified:

### Leaky ReLU

- At z > 0: gradient = ___
- At z < 0: gradient = ___
- How it addresses ReLU's problem:

---

## 3. Vanishing Gradient Analysis

For sigmoid at z = 10:
- Output = ___
- Gradient = ___

Explanation of why this is problematic:



---

## 4. Dead ReLU Analysis

What causes neurons to "die"?



How does Leaky ReLU fix this?



---

## 5. Recommendations

### For Hidden Layers

I would use: _______________

Reasons:
1. 
2. 
3. 

### For Output Layer (Binary Classification)

I would use: _______________

Reasons:
1. 
2. 

### For Output Layer (Multi-Class Classification)

I would use: _______________

Reason:

---

## 6. Key Insight

The most important thing I learned about activation functions is:




---

## 7. Visualization Sketch

Draw rough sketches of each activation function:

```
Sigmoid:          Tanh:             ReLU:
   |  ___            |   ___           |    /
   | /               |  /              |   /
---|/---         ---|----          ---|----
   |                __|              /  |
   |                                /   |
```

