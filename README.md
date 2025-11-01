# ComfyUI Manual Metadata Plugin

A ComfyUI plugin that allows manual input and override of image metadata including prompts, seed, scheduler, steps, model name, and other relevant generation parameters.

## Features

- **Manual Metadata Enhancer**: A node that adds metadata to images before they reach the native SaveImage node, allowing you to enhance or override automatic metadata
- **Manual Save Image**: A drop-in replacement for SaveImage with additional metadata input fields for complete control over saved image metadata

## Installation

1. Clone or download this repository into your ComfyUI `custom_nodes` directory:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/Zorgonatis/comfyui-manual-metadata comfyui-manual-metadata
   ```

2. Restart ComfyUI

3. The new nodes will appear in the node menu under `image/manual_metadata`

## Usage

### Manual Metadata Enhancer Node

This node adds metadata to an image that will be embedded in the image data. Note that ComfyUI's native SaveImage reads metadata primarily from the workflow execution context, so this node embeds metadata directly in the image pixel data using PNG format.

**Important**: Due to how ComfyUI's SaveImage processes images, the embedded metadata may not always be preserved when using the native SaveImage node. For guaranteed metadata preservation, use the Manual Save Image node instead.

To use this node:

1. Connect your image output to the `image` input of the Manual Metadata Enhancer node
2. Fill in any metadata fields you want to add or override:
   - **positive_prompt**: The positive prompt used for generation
   - **negative_prompt**: The negative prompt used for generation
   - **seed**: The random seed value
   - **scheduler**: The scheduler name (e.g., "normal", "karras", etc.)
   - **steps**: Number of inference steps
   - **model_name**: Name of the model used
   - **cfg_scale**: CFG scale value
   - **width**: Image width
   - **height**: Image height
   - **sampler_name**: Name of the sampler used
3. Connect the output to your SaveImage node
4. The metadata will be embedded in the image format

### Manual Save Image Node

This is a drop-in replacement for SaveImage with additional metadata inputs:

1. Connect your image output to the `images` input
2. Set the `filename_prefix` (default: "ComfyUI")
3. Optionally fill in any metadata fields:
   - **positive_prompt**: The positive prompt used for generation
   - **negative_prompt**: The negative prompt used for generation
   - **seed**: The random seed value
   - **scheduler**: The scheduler name
   - **steps**: Number of inference steps
   - **model_name**: Name of the model used
   - **cfg_scale**: CFG scale value
   - **width**: Image width
   - **height**: Image height
   - **sampler_name**: Name of the sampler used
4. The image will be saved with all specified metadata embedded

## Metadata Fields

All metadata fields are optional. Empty strings or default values will be ignored. The metadata is embedded in PNG format using standard PNG text chunks and workflow parameters for ComfyUI compatibility.

### Supported Metadata:

- `prompt` / `positive_prompt`: The positive prompt
- `negative_prompt`: The negative prompt
- `seed`: Random seed value
- `scheduler`: Scheduler type
- `steps`: Number of inference steps
- `model` / `model_name`: Model name
- `cfg_scale`: CFG scale value
- `width`: Image width in pixels
- `height`: Image height in pixels
- `sampler` / `sampler_name`: Sampler name

## Use Cases

- **Manual Metadata Override**: When you want to override automatic metadata detection with specific values
- **External Generation**: When images are generated outside of ComfyUI but you want to add metadata
- **Metadata Enhancement**: When you want to add additional metadata that isn't automatically captured
- **Workflow Documentation**: When you want to ensure specific metadata is always saved with images

## Notes

- The metadata is embedded in PNG format and can be read by tools that support PNG metadata (including ComfyUI's image loader)
- Metadata from Manual Metadata Enhancer is embedded in the image data itself, which should be preserved by SaveImage
- Both nodes support batch processing - all images in a batch will receive the same metadata
- Empty or default values for metadata fields are ignored and won't be added to the output

## Technical Details

The plugin uses PIL/Pillow's `PngImagePlugin` to embed metadata in PNG images. The metadata is stored in standard PNG text chunks and also as a JSON workflow parameter for ComfyUI compatibility.

## License

This plugin is provided as-is. Please check the license file if one is included, or use at your own discretion.

