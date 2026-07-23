---
title: 'Segment Anything 2.1 Tracker'
linkTitle: 'Segment Anything 2.1 Tracker'
weight: 2
description: 'Accelerating video labeling using the SAM2.1 model'
aliases:
  - /docs/enterprise/segment-anything-2-tracker/
---

## Overview

Segment Anything 2.1 is a segmentation model that allows fast and precise selection of any object in videos or images.
SAM2.1 tracking is available in two implementations:

1. **Nuclio SAM2.1 Tracker**: Available for Community self-hosted deployments.
This is implemented as a serverless function deployed via the Nuclio framework.

1. **AI Agent SAM2 Tracker**: Available for CVAT Online and Enterprise via auto-annotation (AA) functions
that run on user-side agents. This brings SAM2 tracking capabilities to CVAT Online users who previously
couldn't access this feature.

It is strongly recommended to deploy the Nuclio model using a GPU. Although it is possible to use a CPU-based version,
it generally performs much slower and is suitable only for handling a single parallel request.
The AI agent variant runs on user hardware, providing flexibility for GPU usage without
server configuration requirements.

Unlike a regular tracking model, both SAM2 tracker implementations are designed to be applied
to existing objects (polygons and masks) to track them forward for a specified number of frames.

## How to install

Choose the installation method based on your platform and deployment needs.

The Community annotation action is registered automatically when the Nuclio function is deployed.

### Nuclio SAM2.1 Tracker (Community self-hosted)

#### Docker

From the CVAT repository root, deploy the GPU function with:

```sh
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam2/nuclio
```

GPU deployment is strongly recommended. CPU deployment is supported but is much slower and suitable only for a
single parallel request.

The tracker requires Redis to store its state between frames. The deployment script configures the function to use
CVAT's Redis state store. Each tracking state expires eight hours after it is created or last updated, so tracking
must finish before the state expires.

#### Kubernetes

- You need to deploy the Nuclio function manually.
Note that this function requires Redis storage configured to keep the tracking state.
You may use the same storage as `cvat_redis_ondisk` uses.
When running the `nuclio deploy` command, make sure to provide the necessary arguments.
The minimal command is:

```sh
nuctl deploy "path/to/the/function"
  --env CVAT_FUNCTIONS_REDIS_HOST="<redis_host>"
  --env CVAT_FUNCTIONS_REDIS_PORT="<redis_port>"
  --env CVAT_FUNCTIONS_REDIS_PASSWORD="<redis_password>" # if applicable
```

### AI Agent SAM2 Tracker (CVAT Online + Enterprise)

The AI agent implementation enables SAM2 tracking for CVAT Online users and provides an alternative deployment method
for Enterprise customers. This approach runs the tracking model on user hardware via auto-annotation (AA) functions.
Deploy SAM2 using Docker Compose with pre-built images for a quick and straightforward setup.

#### Prerequisites

- Docker and Docker Compose
- CVAT Online account or Enterprise instance
- Optional: NVIDIA GPU with CUDA support for faster inference

#### Setup Instructions

The easiest way to deploy SAM2 with pre-built images from Docker Hub.

1. Clone the CVAT repository and navigate to the SAM2 agent directory:
   ```sh
   git clone https://github.com/cvat-ai/cvat.git
   cd cvat/ai-models/agents_deployment/sam2
   ```

1. Create or update the `.env` file with your configuration.
For GPU deployment, set `IMAGE_URL=cvat/sam2_agent:latest_GPU` and `COMPOSE_PROFILES=gpu`.
For CPU-only deployment, set `IMAGE_URL=cvat/sam2_agent:latest` and `COMPOSE_PROFILES=cpu`.
Configure the remaining required variables (`CVAT_BASE_URL`, `CVAT_ACCESS_TOKEN`, `FUNCTION_NAME`, etc.)
following the [corresponding](/docs/guides/compose-agents-userguide/#environment-configuration).

1. Start the agent:
   ```sh
   docker compose up
   ```

1. Verify the agent is running in the CVAT interface.
You should see a new function model named `<FUNCTION_NAME>` in the list on the `/models` page
and in the annotation actions list.

1. To stop and clean up:
   ```sh
   # Deregister the function from CVAT (must be called before volume removed)
   # Alternatively, you can always remove the function from CVAT interface
   docker compose run --rm cvat-function-deregister

   # Stop the agent and remove volumes
   docker compose down -v
   ```

For detailed configuration options and troubleshooting, see the [Docker Compose agent guide](/docs/guides/compose-agents-userguide/).

{{% alert title="Note" color="info" %}}
For enterprise deployments using Kubernetes, refer to the [Docker Compose agent guide](/docs/guides/compose-agents-userguide/)
for Kubernetes deployment instructions and container orchestration examples.
{{% /alert %}}

#### Agent Behavior and Resilience

The AI agent runs as a persistent process on your hardware, providing several advantages:

- **Hardware Independence**: Runs outside the CVAT server, enabling tracking without server-side GPU/Nuclio installation
- **Isolation**: Agent crashes don't affect the server; requests are retried or reassigned automatically
- **Resource Control**: You control the computational resources (CPU/GPU) used for tracking

{{% alert title="Important" color="warning" %}}
Keep the agent process running to handle tracking requests.
If the agent stops, active tracking operations will fail and need to be restarted.
{{% /alert %}}

## Version Requirements

- **AI Agent SAM2 Tracker**: Requires CVAT version 2.42.0 or later
- **Nuclio SAM2.1 Tracker**: Available in Community self-hosted deployments
- **GPU Support**: Optional but recommended for both implementations

## Usage

Both SAM2 tracker implementations provide similar user experiences with slight differences in the UI labels.

### Running the Nuclio SAM2.1 Tracker

The Nuclio tracker can be applied to polygons and masks. Use the existing Segment Anything (SAM) interactor to
create a seed polygon or mask, if needed.
To run the tracker on an object, open the object menu and click
**Run annotation action**.

<img src="/images/sam2_tracker_run_shape_action.png" style="max-width: 200px; padding: 16px;">

Alternatively, you can use a hotkey: select the object and press **Ctrl + E** (default shortcut).
When the modal opens, choose **Segment Anything 2: Tracker** from **Select action**:

<img src="/images/sam2_tracker_run_shape_action_modal.png" style="max-width: 500px; padding: 16px;">

### Running the AI Agent SAM2 Tracker

Once you have registered the SAM2 AI agent and it's running,
you'll see **"AI Tracker: SAM2"** as an available action in the annotation UI for video shape tracking.

To use the AI agent tracker:

1. Create or open a CVAT task from a video file or video-like sequence of images
(all images must have the same dimensions)
1. Open one of the jobs from the task
1. Draw a mask or polygon around an object
1. Right-click the object and choose "Run annotation action"
1. Select **"AI Tracker: SAM2"** from the action list
1. Specify the target frame and click **Run**

The usage flow parallels the existing annotation action interface but utilizes the remote AI agent
rather than built-in serverless functions.

### Tracking Process

Specify the **target frame** until which you want the object to be tracked,
then click the **Run** button to start tracking. The process begins and may take some time to complete.
The duration depends on the inference device, and the number of frames where the object will be tracked.

<img src="/images/sam2_tracker_run_shape_action_modal_progress.png" style="max-width: 500px; padding: 16px;">

Once the process is complete, the modal window closes. You can review how the object was tracked.
If you notice that the tracked shape deteriorates at some point,
you can adjust the object coordinates and run the tracker again from that frame.

## Running on multiple objects

Instead of tracking each object individually, you can track multiple objects
simultaneously. To do this, click the **Menu** button in the annotation view and select the **Run Actions** option:

<img src="/images/sam2_tracker_run_action.png" style="max-width: 200px; padding: 16px;">

Alternatively, you can use a hotkey: just press **Ctrl + E** (default shortcut) when there are no objects selected.
This opens the actions modal. In this case, the tracker will be applied to all visible objects of suitable types
(polygons and masks). In the action list of the opened modal, select either:

- **Segment Anything 2: Tracker** (for the nuclio implementation)
- **AI Tracker: SAM2** (for the AI agent implementation)

<img src="/images/sam2_tracker_run_action_modal.png" style="max-width: 500px; padding: 16px;">

Specify the **target frame** until which you want the objects to be tracked,
then click the **Run** button to start tracking. The process begins and may take some time to complete.
The duration depends on the inference device, the number of simultaneously tracked objects,
and the number of frames where the objects will be tracked.

<img src="/images/sam2_tracker_run_action_modal_progress.png" style="max-width: 500px; padding: 16px;">

Once the process finishes, you may close the modal and review how the objects were tracked.
If you notice that the tracked shapes deteriorate, you can adjust their
coordinates and run the tracker again from that frame (for a single object or for many objects).

## Limitations and Considerations

### AI Agent Limitations

When using the AI agent implementation, keep in mind:

- **Single Agent Constraint**: Only one agent can run at a time for any given tracking function.
Running multiple agents may cause random failures as they compete for tracking states.
- **Memory-based State**: Tracking states are kept in agent memory.
If the agent crashes or is shut down, all tracking states are lost and active tracking processes will fail.
- **Agent-only Usage**: Tracking functions can only be used via agents.
There is no equivalent of the `cvat-cli task auto-annotate` command for tracking.
- **Rectangle Limitation**: When using the AI Tools dialog (sidebar),
only tracking functions that support rectangles will be selectable.
The SAM2 tracker supports polygons and masks but not rectangles.
- **Skeleton Tracking**: Skeletons cannot currently be tracked by either implementation.


## Tracker parameters

- **Target frame**: Objects will be tracked up to this frame. Must be greater than the current frame
- **Convert polygon shapes to tracks**: When enabled, all visible polygon shapes in the current frame will be converted
to tracks before tracking begins. Use this option if you need tracks as the final output but started with shapes,
produced for example by interactors (e.g. SAM2 or another one).

Polygon seeds can become tracks when this option is enabled. Masks remain per-frame shapes.

## See Also

- [SAM2 Object Tracking via AI Agent (Blog, July 31, 2025)](https://www.cvat.ai/resources/blog/sam2-ai-agent-tracking) -
Detailed implementation and background information
- [Auto-annotation Functions Documentation](https://docs.cvat.ai/docs/api_sdk/sdk/auto-annotation/) -
Reference for creating custom tracking functions
- [CVAT CLI Examples](https://docs.cvat.ai/docs/api_sdk/cli/#examples---functions) - Additional CLI usage examples
