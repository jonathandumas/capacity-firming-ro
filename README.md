# Capacity-firming
Official implementation of a two-stage robust optimization algorithm in the capacity firming frameworkand the experiments presented in the paper:
> Dumas, Jonathan, et al. "Probabilistic forecasting for sizing in the capacity firming framework." arXiv preprint arXiv:2106.02323 (2021).
> [[arxiv]](https://arxiv.org/abs/2106.02323)

Note: this paper is under review for IEEE-TSE.

The extended version of this paper is:
> TODO
> [[arxiv]]()

## Cite

If you make use of this code in your own work, please cite our arXiv paper:

```
@article{dumas2021probabilistic,
  title={A Probabilistic Forecast-Driven Strategy for a Risk-Aware Participation in the Capacity Firming Market},
  author={Dumas, Jonathan and Cointe, Colin and Wehenkel, Antoine and Sutera, Antonio and Fettweis, Xavier and Corn{\'e}lusse, Bertrand},
  journal={arXiv preprint arXiv:2105.13801},
  year={2021}
}
```

Note: the reference will be changed if the paper is accepted for publication in IEEE-TSE.

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
It allows to configure the capacity firming parameters such as the tolerance deviation, etc.
By default it is set to the parameters used for the paper.

## Data
The ro/data folder contains the ULiège dataset, the day-ahead PV quantile forecasts of the LSTM and NF models, the day-ahead PV point forecasts, and the PV intraday point forecasts.

## Determinist optimization 
The ro/determinist folder contains the file:
```bash
det_loop.py 
```
It allows to compute the day-ahead planning using the deterministic planner (MILP) and to compute the intraday dispatch with the controller (MILP).
The planner and controller are located into the folder ro/determinist/algorithms.

## Robust optimization 

The ro/robust folder contains the file:
```bash
ro_loop.py 
```
It allows to compute the day-ahead planning using the robust algorithm (BD or CCG). Note: the choice of the algorithm must be specified into the file. In addition, it is possible to make the static or dynamic robust optimization (cf the paper for more details), and to select the LSTM or NF PV quantile forecasts. You have to specify the risk-averse parameters for each mode. By default, it computes the day-ahead planning over the testing set that is composed of 30 days randomly selected into the dataset.
The default mode is the CCG algorithm using the NF PV quantile forecasts in static robust optimization with the risk-averse parameters: q = 10 % and gamma = 12.

The file:
```bash
controller_loop.py 
```
allows to compute the intraday dispatch using the day-ahead planning computed by the robust algorithm (BD or CCG).

The ro/robust/results folder contains the file:
```bash
res_heatmap.py 
```
It allows to draw the heatmaps presented in the paper to compte the results.

The ro/robust/algorithms folder contains the BD and CCG algorithms.

The procedure to use the code is the following:
* specify in the `ro_loop.py` file the algorithm (BD or CCG), the type of robust optimization (static or dynamic), and the numerical parameters of the algorithm (convergence threshold, big-M's values, etc);
* run `ro_loop.py` -> the day-ahead planning are computed;
* run `controller_loop.py` -> the intraday profits are computed by the controller that uses the intraday PV point forecasts and the day-ahead plannings
* when you have computed all the day-ahead planning for all pairs of risk-averse parameters for both the BD and CCG algorithms for both the LSTM and NF PV quantiles you can use `res_heatmap.py`. Note: you have to compute the day-ahead planning in the deterministic mode also with `det_loop.py` in the ro/determinist to have the entire heatmaps.

