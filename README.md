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

The project includes a small development web UI for quickly uploading images
and previewing gravitational lensing with a black hole placed at the image
center. Features:

- Upload images (stored in an `uploads/` folder created next to the repo root).
- Delete uploaded images (small ✖ button on each thumbnail).
- Preview rendering using two methods:
  - Weak-field: fast per-pixel angular shift using Einstein's weak-field
    approximation (default).
  - Geodesic: slower, more accurate coarse numeric geodesic tracing (per-radial
    bin integration, interpolated across the image). Slower but visually more
    accurate for stronger lensing.

How to run the web preview locally

1. Ensure Flask and Pillow are installed (either via `uv sync` or the
   virtualenv example above).

2. Start the development server. The app honors the `PORT` environment
   variable; if not set it defaults to `8000`:

```sh
# run on default port 8000
python3 -m gravity_mirage.web

# or set an explicit port
PORT=5000 python3 -m gravity_mirage.web
```

In your browser open `http://localhost:8000` (or the port you chose).

Usage notes

- Click a thumbnail to set it as the active preview image, or select an image
  from the dropdown in the "Black hole preview" panel.
- Choose mass (in solar masses) and scale (how many Schwarzschild radii the
  image radius corresponds to) and pick "Weak-field" or "Geodesic" method.
- Click "Render preview" to request a generated PNG of the lensed image.

Performance and recommendations

- Weak-field mode is fast and suitable for interactive previews.
- Geodesic mode performs numeric integration for a coarse set of impact
  parameters (default limited to keep response times reasonable). Use smaller
  output widths (e.g., 256) for faster previews, and increase resolution or
  bin count if you need higher fidelity.
- For longer or heavy renders consider adding an asynchronous job queue or
  running the renderer in a background worker — I can help add that next.
- The web preview is intended for local development and testing; it is not hardened for production use.
