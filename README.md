# Capacity-firming
Official implementation of a two-stage robust optimization algorithm in the capacity firming framework and the experiments presented the paper:
- J. Dumas, C. Cointe, A. Wehenkel, A. Sutera, X. Fettweis and B. Cornelusse, "A Probabilistic Forecast-Driven Strategy for a Risk-Aware Participation in the Capacity Firming Market," in IEEE Transactions on Sustainable Energy, doi: 10.1109/TSTE.2021.3117594.

## Cite

If you make use of this code, please cite our IEEE paper:

```
@article{dumas2021probabilistic,
  author={Dumas, Jonathan and Cointe, Colin and Wehenkel, Antoine and Sutera, Antonio and Fettweis, Xavier and Cornelusse, Bertrand},
  journal={IEEE Transactions on Sustainable Energy}, 
  title={A Probabilistic Forecast-Driven Strategy for a Risk-Aware Participation in the Capacity Firming Market}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TSTE.2021.3117594}}
```

Note: an extended version is available on [arXiv](https://arxiv.org/abs/2105.13801)

# Forecast-driven robust optimization strategy
![strategy](https://github.com/jonathandumas/capacity-firming/blob/9a54f129fa67d4be094d076c8e29dd2f5eaca3be/figures/methodology_scheme.png)

# Dependencies
The Python Gurobi library is used to implement the algorithms in Python 3.7, and [Gurobi](https://www.gurobi.com/) 9.0.2 to solve all the optimization problems.

# Usage
## Configuration of the capacity firming parameters
In the ro folder is the file:
```bash
ro_simulator_configuration.py 
```
It allows configuring the capacity firming parameters such as the tolerance deviation, etc.
By default, it is set to the parameters used for the paper.

## Data
The ro/data folder contains the ULiège dataset, the day-ahead PV quantile forecasts of the LSTM and NF models, the day-ahead PV point forecasts, and the PV intraday point forecasts.

## Determinist optimization 
The ro/determinist folder contains the file:
```bash
det_loop.py 
```
It allows to compute the day-ahead planning using the deterministic planner (MILP) and to compute the intraday dispatch with the controller (MILP).
The planner and controller are located in the folder ro/determinist/algorithms.

## Robust optimization 

The ro/robust folder contains the file:
```bash
ro_loop.py 
```
It allows computing the day-ahead planning using the robust algorithm (BD or CCG). Note: the choice of the algorithm must be specified in the file. In addition, it is possible to make the static or dynamic robust optimization (cf the paper for more details) and select the LSTM or NF PV quantile forecasts. The risk-averse parameters have to be specified for each type. By default, it computes the day-ahead planning over the testing set composed of 30 days randomly selected into the dataset.
The default mode is the CCG algorithm using the NF PV quantile forecasts in robust static optimization with the risk-averse parameters: q = 10 % and gamma = 12.

The file:
```bash
controller_loop.py 
```
allows computing the intraday dispatch using the day-ahead planning computed by the robust algorithm (BD or CCG).

The ro/robust/results folder contains the file:
```bash
res_heatmap.py 
```
It allows drawing the heatmaps presented in the paper to compute the results.

The ro/robust/algorithms folder contains the BD and CCG algorithms.

The procedure to use the code is the following:
* specify in the `ro_loop.py` file the algorithm (BD or CCG), the type of robust optimization (static or dynamic), and the numerical parameters of the algorithm (convergence threshold, big-M's values, etc);
* run `ro_loop.py` -> the day-ahead planning are computed;
* run `controller_loop.py` -> the intraday profits are computed by the controller that uses the intraday PV point forecasts and the day-ahead plannings
* `res_heatmap.py` is used to draw the heat maps depicted below. Note: the day-ahead planning in the deterministic mode must be computed with `det_loop.py` in the ro/determinist to get the complete heatmaps.

## Example: NF-BD static robust optimization
For instance, by using the BD algorithm in robust static optimization for several pairs of risk-averse parameters:
![BD-NF-static](https://github.com/jonathandumas/capacity-firming-ro/blob/23ac75ae0a626fdfa4b65ccffd9a99e0a9998020/figures/BD_NF_RO_static.png)
* Nominal is the deterministic day-ahead planner using PV point forecasts;
* the column \ is the deterministic day-ahead planner using PV quantile forecasts;
* the BD algorithm computed day-ahead planning for several pairs (q, gamma);

## Example: NF-CCG dynamic, robust optimization
For instance, by using the CCG algorithm in dynamic, robust optimization for several pairs of risk-averse parameters:
![CCG-NF-dynamic](https://github.com/jonathandumas/capacity-firming-ro/blob/1c4ef372746705624518984da4eed19d0eb2ea63/figures/CCG_NF_RO_dyn.png)
* Nominal is the deterministic day-ahead planner using PV point forecasts;
* the column \ is the deterministic day-ahead planner using PV quantile forecasts;
* the BD algorithm computed day-ahead planning for several pairs (d_q, d_gamma);
