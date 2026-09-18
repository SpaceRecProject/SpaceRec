# H&E / SpaceRec image comparison

Public site: <https://spacerecproject.github.io/SpaceRec/>

## Single-file offline version

`SpaceRec_image_comparison_standalone.html` contains the application and registered example images in one file. Download it and double-click it in a current Chrome, Safari, Firefox, or Edge browser. No server or installation is required.

## Directory version

Run from the project root:

```bash
python3 -m http.server 8765 --directory resources/dbitplus/model_pred/image_comparison
```

Open `http://localhost:8765`.

When the server is running on a Bridges2 compute node, forward it from a Mac terminal with:

```bash
ssh -N -L 8766:<COMPUTE_NODE>:8765 gwang17@bridges2.psc.edu
```

Then open `http://localhost:8766` on the Mac.

The registered DBiTplus H&E, SpaceRec, and Clusters 0-4 images are bundled under `assets/`. Choose H&E/SpaceRec, H&E/Clusters, or SpaceRec/Clusters in the toolbar. Color legends are displayed outside the tissue image and included in PNG exports. The H&E and SpaceRec selectors can replace those layers with local images. Exact swipe/opacity alignment requires both active images to use the same coordinate bounds and aspect ratio.
