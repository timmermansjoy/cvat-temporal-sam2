# SAM3.1 text/point interactor and multiplex tracker

A Nuclio runner for text, box and point prompts, plus tracking of CVAT masks
and polygons. It uses Meta's SAM3.1 detector, text encoder and Object Multiplex
tracker with one shared vision backbone. SAM2 Tiny remains available as the
default tracker.

## Deploy

1. Obtain access to [Meta's SAM3.1 checkpoint](https://huggingface.co/facebook/sam3.1)
   and download `sam3.1_multiplex.pt` to this path **on the Docker host**:

   ```text
   /opt/cvat/models/sam3.1/sam3.1_multiplex.pt
   ```

   The function mounts this directory read-only. Change `spec.volumes` and
   `SAM31_CHECKPOINT` in `function-gpu.yaml` if using another location. Checkpoint
   access tokens and model weights are not embedded in the image.

2. Start CVAT with its Nuclio/serverless services and NVIDIA Container Toolkit,
   then deploy from the repository root:

   ```sh
   ./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam3_1/nuclio
   ```

   The image installs Python 3.12 and PyTorch 2.10.0 / CUDA 12.8. The NVIDIA host
   driver must support that CUDA runtime. The runner requires CUDA and BF16
   support; there is no CPU inference configuration.

3. Rebuild the CVAT UI with these changes. In the standard annotation workspace,
   select **SAM3.1** beside the frame-count dropdown. The model is enabled in the
   dropdown once CVAT discovers the deployed function.

4. Create masks using **AI tools → Interactors → Segment Anything 3.1: Text and points**
   (see below), the existing SAM2 interactor, or manual drawing. Use **G/S** for the
   selected object or **Shift+G/Shift+S** for all visible unlocked masks/polygons.
   Frame count, job boundaries, undo, and cancellation work through the existing
   tracker action. Alternatively select **Segment Anything 3.1: Tracker** in
   **Run annotation action**.

## Text and point annotation

1. Choose the SAM3.1 interactor and the CVAT label to assign to the results.
2. Enter one short description, such as `person` or `red car`, and click **Find objects**.
   This searches the current frame and previews up to 64 matching masks. With an ROI,
   it searches only that region. Nothing is added to the annotations yet.
3. Review the masks and use the confidence slider to filter detections.
   To correct one mask, left-click inside it first, then use positive/negative
   clicks for that object. Other detected masks remain in the preview. A first
   positive click outside the detections starts a new object instead.
4. Press **Enter** to accept the visible masks or **Esc** to discard the preview.
   To refine another object, accept the results and use the existing mask refinement
   action on that object. Delete false detections before propagation.
5. Use **Shift+G / Shift+S** to propagate the accepted masks within the current job.

Leaving the text field empty retains point-based mask creation. The optional box
prompt provides a visual example for detecting matching objects. Refining an
existing mask ignores the text field and uses points for that single object.
Each request includes the complete prompt and recomputes the current frame;
interaction previews do not change Redis tracking states. Text is limited to 256
characters and each positive/negative point list to 256 points.

The detector searches the prompted frame. Shortcut propagation tracks the masks
you accepted; it does not run text detection for newly appearing objects on later
frames. Run another text search when you need to add those objects.

## Runtime behavior

- The upstream source is pinned to
  [`2345a4ad109ac29c569da749c91d84f10dc08c40`](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40).
  The adapter uses `build_sam3_multiplex_video_predictor` and `add_prompt` for
  interaction, and the same model's tracker with `init_state`, `add_new_masks`,
  and `propagate_in_video` for existing masks. All checkpoint weights are checked
  before serving requests. FlashAttention 3 and compilation
  are disabled. The multiplex bucket size is the checkpoint's default of 16.
- Up to 64 objects and 10 images per HTTP request. The UI sends longer runs as
  successive requests. Backward tracking feeds frames in reverse order.
- State is stored in CVAT's Redis under `cvat:sam3.1:state:` with an eight-hour
  lifetime. Every successful request creates a new immutable snapshot. Retrying
  or cancelling a request does not advance its input state.
- Corrected objects can have independent seed states. The runner groups requests
  by seed state and preserves the caller's object ordering. Objects from each
  shared seed are still processed jointly. Starting a fresh run with all objects
  seeds them together again.
- Video tensors and image features are not persisted. Temporal memory retains
  the seed frame and the recent frames needed by the tracker. State is offloaded
  to CPU between inference operations.
- One worker uses one GPU. Multiple cards do not pool VRAM. This change has not
  been benchmarked on the RTX 3060 or RTX 5080; start with a short run and a few
  objects, and compare memory use and correction effort with SAM2 Tiny. The full
  detector/text model needs more memory than the previous tracker-only runner.

## Checks

Request, state, mask, and failure tests run without a GPU or model checkpoint:

```sh
python -m unittest discover \
  -s serverless/pytorch/facebookresearch/sam3_1/nuclio -p test_main.py -v
```

They require `torch`, `numpy`, `Pillow`, `opencv-python-headless`, and `redis`.
The model is stubbed in these tests; they do not verify inference quality or
checkpoint compatibility on a CUDA host.

UI request, preview, acceptance, cancellation and SAM2 compatibility checks:

```sh
node cvat-ui/tests/sam31-prompts.cjs
```

After deployment, the existing tracker benchmark can evaluate SAM3.1 using
human reference masks without a click interactor:

```sh
python serverless/pytorch/facebookresearch/sam2/nuclio/benchmark.py \
  --endpoint http://localhost:<sam3-function-port> \
  --references-json references.json frame000.png frame001.png frame002.png
```

`references.json` contains one CVAT RLE mask per input image, in image order.
Also smoke-test two objects with Shift+G and Shift+S, edit one prediction and
propagate again, and cancel a longer run. For interaction, test text-only results,
positive/negative clicks on one result, an ROI, confidence filtering, Enter, and
Escape while inference is running. These GPU checks are required before treating
this new runner as validated for annotation.
