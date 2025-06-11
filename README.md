# SAE Feature Consistency

This repository contains the code and experiments for our paper on prioritizing feature consistency in Sparse Autoencoders (SAEs) for mechanistic interpretability.

**📄 Paper:** [Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs](https://arxiv.org/abs/2505.20254)

**👥 Authors:** Xiangchen Song*, Aashiq Muhamed*, Yujia Zheng, Lingjing Kong, Zeyu Tang, Mona T Diab, Virginia Smith, Kun Zhang  
*Equal contribution

## 🚀 Quick Start

### Environment Setup

This codebase uses [`uv`](https://github.com/astral-sh/uv) as the package manager. To set up the environment:

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Repository Structure

- **[`synthetic/`](./synthetic/)** - Synthetic data experiments demonstrating feature consistency principles
- **[`examples/`](./examples/)** - Experiments with real LLM activations

## 🙏 Acknowledgments

This work builds upon excellent open-source projects:

- **[dictionary_learning_demo](https://github.com/adamkarvonen/dictionary_learning_demo)** by Adam Karvonen
- **[dictionary_learning](https://github.com/saprmarks/dictionary_learning)** by Samuel Marks, Adam Karvonen, and Aaron Mueller

We thank the authors for making their code available and enabling this research.

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{song2025position,
  title={Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs},
  author={Song, Xiangchen and Muhamed, Aashiq and Zheng, Yujia and Kong, Lingjing and Tang, Zeyu and Diab, Mona T and Smith, Virginia and Zhang, Kun},
  journal={arXiv preprint arXiv:2505.20254},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0. See the [`LICENSE`](./LICENSE) file for details.

## 📞 Contact

For questions or issues, please [open an issue](https://github.com/xiangchensong/sae-feature-consistency/issues) on this repository.

