# Neural Curve Fitting

A Conditional Neural Process implemented in PyTorch. The model is trained on a set
of functions rather than a single one, so given a few observed points from an unseen
curve it must identify which function they belong to before predicting the rest of it.

## The problem

The training set contains 30,000 functions, each with 100 points on the x-axis over
[-1, 1]. For both training and testing, a random subset of N observed pairs
(x_o, y_o) is provided as prior information about which function the model is
looking at.

This means the model has to perform two tasks: function selection, choosing which
function from the set it needs to model, and regression, learning the input-output
relationship for that function.

## Model architecture

The model consists of two MLPs, an Encoder and a Decoder, which are trained jointly
through a wrapper Model class.

The Encoder takes a set of N observed (x_o, y_o) pairs and maps each pair to a
feature vector of size `r_dim`. These feature vectors are then averaged into a
single summary vector r_O, which acts as a compressed representation of the curve.

The Decoder takes r_O and a set of target values x_t and predicts the corresponding
y_t values. r_O is repeated for each target point and concatenated with x_t before
being passed through the MLP, giving an input size of `r_dim + 1`.

Both MLPs have two hidden layers of size `h_dim` with ReLU activations, and no
activation on the final output layer, consistent with a regression task.

## Training

The training function jointly optimises the Encoder and Decoder using MSE loss and
the Adam optimiser. Each iteration, N context points are randomly sampled (between
`n_min` and `n_max`) from each function in the batch; this makes the model robust to
varying numbers of observed points at test time. All 100 points are used as targets.

## Data

The datasets were provided with the module and are not included here. `data/`
expects three files:

| File                   | Contents                              |
| ---------------------- | ------------------------------------- |
| `train_data.csv`       | 30,000 rows, 100 y values per row     |
| `test_data.csv`        | 300 rows, same format                 |
| `test_observations.pt` | fixed context sets used at evaluation |

x values are not stored, they are `linspace(-1, 1, 100)` and reconstructed in
`dataset.py`.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

This trains three models at `r_dim` 2, 4 and 8, evaluates each across the five
context schemes in `test_observations.pt`, and prints the correlation matrix for the
latent representation of model4.

## Results

All three models were trained for 40 epochs with `h_dim=128`, `lr=0.001`, and
context points sampled randomly between 3 and 20. Each model was evaluated across 5
context schemes from `test_observations.pt`.

Test MSE results:

| Context Scheme | model2 (r_dim=2) | model4 (r_dim=4) | model8 (r_dim=8) |
| -------------- | ---------------- | ---------------- | ---------------- |
| 10_stride1     | 0.378502         | 0.325101         | 0.329297         |
| 20_stride1     | 0.333744         | 0.268500         | 0.270210         |
| 20_stride2     | 0.258324         | 0.162209         | 0.155150         |
| 10_even        | 0.040977         | 0.010808         | 0.008362         |
| 20_even        | 0.040331         | 0.010457         | 0.007939         |

All models performed significantly better on evenly spaced context schemes (10_even,
20_even) compared to stride-based schemes, suggesting the spatial distribution of
context points has a large impact on performance. This is expected as evenly spaced
points cover the full input range [-1, 1], giving the Encoder information about the
entire curve. Stride-based points are clustered at one end, forcing the model to
extrapolate across the rest of the curve, which is more difficult for the model.

### Effect of latent dimension

model8 yielded the best overall results across all evaluation schemes. model2 showed
notably weaker performance across all schemes, indicating `r_dim=2` lacks the
capacity to fully represent the variety of curves in the dataset. model4 and model8
performed comparably on stride-based schemes, with model4 marginally outperforming
model8 on 10_stride1 (0.325 vs 0.329) and 20_stride1 (0.269 vs 0.270), suggesting
that a larger latent representation does not always guarantee better performance.
model8 outperformed model4 noticeably on even context schemes.

The most significant improvement came from increasing `r_dim` from 2 to 4. model4
outperformed model2 substantially across every context scheme, whereas the
improvement from model4 to model8 was much smaller. On stride-based schemes model8
actually performed marginally worse than model4.

### Number of independent variables

The data was generated using 4 independent variables. With `r_dim=2`, the latent
representation does not have enough dimensions to encode all the underlying
variables that define each curve, limiting the model's ability to distinguish
between functions. With `r_dim=4`, the model gains enough capacity to represent all
4 variables, resulting in a large performance improvement. The smaller improvement
from model4 to model8 supports this, as the extra dimensions in model8 are not
needed since model4 already has enough capacity to represent all 4 independent
variables.

### Disentanglement of the latent representation

To assess how well model4 disentangles the independent variables, the latent
representations r_O were collected for all 300 test functions using the 20_even
context scheme. The correlation matrix between the 4 dimensions of r_O was then
computed:

|        | r1     | r2     | r3     | r4     |
| ------ | ------ | ------ | ------ | ------ |
| **r1** | 1.000  | -0.916 | 0.018  | 0.310  |
| **r2** | -0.916 | 1.000  | 0.087  | -0.501 |
| **r3** | 0.018  | 0.087  | 1.000  | -0.821 |
| **r4** | 0.310  | -0.501 | -0.821 | 1.000  |

The results show a mix of correlated and uncorrelated dimension pairs. Some pairs
are strongly correlated (r1 and r2 at -0.916, r3 and r4 at -0.821), while others are
close to 0 (r1 and r3 at 0.018, r1 and r4 at 0.310), suggesting only partial
disentanglement.

The strong correlations indicate that some dimensions are capturing combinations of
the underlying independent variables rather than one variable each. This is likely
because the model was only trained to minimise prediction error, with no incentive
to keep the dimensions of r_O independent. As a result, the model learns whatever
representation works best for reconstructing the curve, which does not necessarily
mean each dimension corresponds to a separate independent variable.

Overall, model4 has learnt a useful latent representation for curve prediction but
has only partially separated the 4 independent variables into distinct dimensions.

## Files

```
dataset.py    loads the datasets and builds the dataloader
model.py      Encoder, Decoder and the combined Model
train.py      training function and hyperparameters
evaluate.py   evaluation across the test context schemes
main.py       trains all three models, evaluates them, analyses the latents
```

## Background

Written for the Neural Networks and Deep Learning module at Queen Mary University of
London.
