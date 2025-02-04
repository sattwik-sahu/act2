# Action Chunking Transformer 2

*Sattwik Sahu, IISER Bhopal*

This is a slightly modified implementation of the Action Chunking Transformers paper by Zhao et al.

## Usage

### Clone the repo

```bash
git clone https://github.com/sattwik-sahu/act2.git
cd act2
```

### Installation and Setup

1. Requires Python version `>= 3.12`
2. Install the `uv` Python dependency manager tool using
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3. Install Python using
    ```bash
    uv python install 3.12
    ```
4. Install and sync all dependencies using
    ```bash
    uv sync
    ```

## Files

### Models

**Directory:** [`src/utils/models/`](src/utils/models/)

**Purpose:** Contains files that determine the actual architecture of the model being implemented. Modularizes and abstracts the implementation of the architecture to make it easy to introduce new *plug 'n play* modules in the architecture (*e.g.* Switching thee ResNet18 image encoder for a different one).

1. [`src/utils/models/stylevar.py`](src/utils/models/stylevar.py)
    contains code for the **style variable encoder**. It takes an action sequence and the joint states and outputs a style variable $z$.
2. [`src/utils/models/image_encoder.py`](src/utils/models/image_encoder.py)
    contains code for an abstract **image encoder** module. It takes in an image as a `torch.Tensor` of dimensions $H \times W \times C$ and output s whatever the output dimensions are, based on the type of backbone. Also has code for a `ResNet18` image encoder backbone.
3. [`src/utils/models/action_transformer.py`](src/utils/models/action_transformer.py)
    contains the code for the **encoder and decode modules** of the action chunking transformer.

> The modules discussed in \[2\] and \[3\] are contenders for the modifications planned to the ACT architecture to make it work for natural language instructions.

### Data

**Directory:** [`src/utils/data/`](src/utils/data/)

**Purpose:** Contains utilities to load datasets and use them, also define custom data structures if required, for the loaded data, transforms to be applied on them, etc.

---

> 🚧 This repo is under construction
