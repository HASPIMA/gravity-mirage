# Gravity Mirage

Simulator that visually demostrates gravitational lensing caused by
black holes.

## Table of contents

- [Gravity Mirage](#gravity-mirage)
  - [Table of contents](#table-of-contents)
  - [Running and setting up the project](#running-and-setting-up-the-project)
    - [Local development](#local-development)
      - [Setup locally](#setup-locally)
      - [Running the project](#running-the-project)
    - [Web preview (browser)](#web-preview-browser)

## Running and setting up the project

### Local development

#### Setup locally

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

1. Install required packages

    ```sh
    uv sync
    ```

    The project's `pyproject.toml` includes the runtime dependencies (numpy, scipy,
    matplotlib, pygame) and the small web preview dependencies (Flask, Pillow).

    Alternatively, for a lightweight developer environment you can create a
    virtualenv and install only what you need:

    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install Flask Pillow numpy scipy
    ```

#### Running the project

Once you have done [the setup step](#setup-locally), you may run the
project by executing (command-line entrypoint):

```sh
uv run gravity-mirage
```

### Web preview (browser)

The repository includes a small development web UI for uploading images and
previewing gravitational lensing. This UI is implemented with FastAPI and the
server is served with Uvicorn (not Flask). The app creates an `uploads/`
directory next to the repository root when needed and uses two rendering
methods:

- Weak-field: fast per-pixel angular deflection using Einstein's weak-field
  approximation (interactive preview).
- Geodesic: slower, more accurate numeric geodesic tracing (coarse radial
  integration, interpolated across the image).

Important requirements

- Python 3.13+ (the project's `pyproject.toml` requires >= 3.13).
- Use a virtual environment for local development (recommended).

How to run the web preview locally (recommended)

1. Create and activate a virtual environment in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package (recommended) which will pull runtime dependencies including FastAPI and Uvicorn

```bash
uv sync
```

Start the development server with auto-reload (from an activated venv):

```bash
# if venv is active
python -m gravity_mirage.web:app

# or explicitly using the venv binary
.venv/bin/uvicorn gravity_mirage.web:app
```

In your browser open `http://127.0.0.1:8000` (or the port you chose).

Usage notes

- Upload images from the left-hand panel; uploaded files are saved into
  `uploads/` next to the repo root.
- Click a thumbnail to set it as the active preview image or choose one from
  the dropdown in the preview panel. The preview image automatically updates
  when you change parameters (mass, scale, method).
- The UI requests generated PNG previews from the `/preview/{filename}`
  endpoint; no separate "Render" button is required.

Performance and recommendations

- Use Weak-field mode for fast, interactive previews. Use Geodesic mode for
  higher fidelity at greater computational cost.
- Reduce the preview width (e.g., 256) to get faster responses from the
  geodesic renderer during iteration.
- This preview server is intended for development and testing. For production
  use add appropriate hardening (reverse proxy, auth, rate limiting,
  background workers, etc.).

Troubleshooting

- If `uvicorn` is not found, ensure your virtualenv is activated and that
  dependencies are installed inside it (`source .venv/bin/activate` and
  `python -m pip install -e .`).
- Verify the binary and version:

```bash
which uvicorn
uvicorn --version
python -c "import uvicorn; print(uvicorn.__version__)"
```

- If imports fail when starting the server, install the missing package into
  the active venv with `python -m pip install <package>`.

If you want, I can add a small section with a one-line development-startup
script or a tiny systemd unit for running the preview on a server.
