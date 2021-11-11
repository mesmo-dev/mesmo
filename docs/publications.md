# Publications

## Citing MESMO

The recommended way to cite MESMO is to refer to its latest [Zenodo record](https://doi.org/10.5281/zenodo.3523568). You can use the following bibtex entry:

```{literalinclude} ../CITATION.bib
:language: bib
```

## MESMO-enabled publications

The following publications are based on MESMO.

- [Preprint] Zhang, K, Troitzsch, S., Han, X., **Distributionally Robust Co-optimized Offering for Transactive Multi-energy Microgrids**, 2021.
    - MESMO [v0.5.0](https://github.com/mesmo-dev/mesmo/releases/tag/v0.5.0) was used to prepare the results for this paper. The related scripts are at `examples/publications/paper_2021_zhang_distributionally_robust_optimization`.
- [Preprint] Kleinschmidt, V., Troitzsch, S., Hamacher, T., Perić, V., **Flexibility in distribution systems – Modelling a thermal-electric multi-energy system in MESMO**, 2021.
    - MESMO [v0.4.0](https://github.com/mesmo-dev/mesmo/releases/tag/v0.4.0) was used to prepare the results for this paper. The related script is at `examples/publications/paper_2021_kleinschmidt_isgt_multi_energy_system.py`.
- [Preprint] Troitzsch, S., Zhang, K., Massier, T., & Hamacher, T., **Coordinated Market Clearing for Combined Thermal and Electric Distribution Grid Operation**, in IEEE Power & Energy Society General Meeting (PESGM), 2021. [`doi:10.36227/techrxiv.13247246`](https://doi.org/10.36227/techrxiv.13247246)
    - MESMO [v0.4.0](https://github.com/mesmo-dev/mesmo/releases/tag/v0.4.0) was used to prepare the results for this paper. The related script is at `examples/publications/paper_2021_troitzsch_admm_coordination_thermal_electric.py`.
- Troitzsch, S., Grussmann, M., Zhang, K., & Hamacher, T., **Distribution Locational Marginal Pricing for Combined Thermal and Electric Grid Operation**, IEEE PES Innovative Smart Grid Technologies Conference Europe, 2020. [`doi:10.1109/ISGT-Europe47291.2020.9248832`](https://doi.org/10.1109/ISGT-Europe47291.2020.9248832)
    - MESMO [v0.3.0](https://github.com/mesmo-dev/mesmo/releases/tag/v0.3.0) was used to prepare the results for this paper. The related script is at `examples/publications/paper_2020_dlmp_combined_thermal_electric.py`.

## Foundational publications

The following publications served as the methodological basis for the implementation of MESMO as well as for the development of test cases.

- Troitzsch, S., Hanif, S., Zhang, K., Trpovski, A., & Hamacher, T., **Flexible Distribution Grid Demonstrator (FLEDGE): Requirements and Software Architecture**, in IEEE PES General Meeting, Atlanta, GA, USA, 2019. [`doi: 10.1109/PESGM40551.2019.8973567`](https://doi.org/10.1109/PESGM40551.2019.8973567).
    - This paper served as an outline for the software architecture of MESMO.
- D. Recalde, A. Trpovski, S. Troitzsch, K. Zhang, S. Hanif, and T. Hamacher, **A Review of Operation Methods and Simulation Requirements for Future Smart Distribution Grids**, in IEEE PES Innovative Smart Grid Technologies Conference Asia, Singapore, 2018. [`doi:10.1109/ISGT-Asia.2018.8467850`](https://doi.org/10.1109/ISGT-Asia.2018.8467850).
    - This review paper initiated the development of MESMO.
- S. Hanif, K. Zhang, C. Hackl, M. Barati, H. B. Gooi, and T. Hamacher, **Decomposition and Equilibrium Achieving Distribution Locational Marginal Prices using Trust-Region Method**, IEEE Transactions on Smart Grid, 2018. [`doi:10.1109/TSG.2018.2822766`](https://doi.org/10.1109/TSG.2018.2822766).
    - This paper outlines the mathematical definitions for the linear global-approximate electric grid modeling method in MESMO.
- A. Trpovski, D. Recalde, and T. Hamacher, **Synthetic Distribution Grid Generation Using Power System Planning: Case Study of Singapore**, in UPEC International Universities Power Engineering Conference, 2018. [`doi:10.1109/UPEC.2018.8542054`](https://doi.org/10.1109/UPEC.2018.8542054).
    - This paper outlines the synthetic grid model that was used with MESMO for Singapore-based studies at TUMCREATE. However, the complete synthetic test case is currently not part of this repository.
