import os
import json
from PIL import Image, PngImagePlugin
import folder_paths


def get_model_name(model):
    """Extract model name from MODEL type object."""
    if model is None:
        return ""
    # Try to get model name from various possible attributes
    if hasattr(model, 'model_path'):
        return os.path.basename(model.model_path) if model.model_path else ""
    if hasattr(model, 'model_file'):
        return os.path.basename(model.model_file) if model.model_file else ""
    if hasattr(model, 'name'):
        return model.name if model.name else ""
    # Last resort: try to get from any path-like attribute
    for attr in ['path', 'filename', 'file_path', 'checkpoint_path']:
        if hasattr(model, attr):
            path = getattr(model, attr, None)
            if path:
                return os.path.basename(path)
    return ""


def get_prompt_from_conditioning(conditioning):
    """Try to extract prompt text from CONDITIONING type object."""
    if conditioning is None:
        return ""
    # CONDITIONING is typically a list of tuples or dicts
    # Try to find prompt text in various possible locations
    if isinstance(conditioning, (list, tuple)):
        for item in conditioning:
            if isinstance(item, dict):
                # Check common locations for prompt text
                for key in ['prompt', 'text', 'positive', 'negative', 'conditioning_prompt']:
                    if key in item and isinstance(item[key], str):
                        return item[key]
                # Check nested structures
                if 'pooled_output' in item or 'conditioning' in item:
                    # Look for text in nested dict
                    nested = item.get('pooled_output') or item.get('conditioning')
                    if isinstance(nested, dict):
                        for key in ['prompt', 'text', 'positive', 'negative']:
                            if key in nested and isinstance(nested[key], str):
                                return nested[key]
    return ""


class ManualMetadataEnhancer:
    """
    Node that adds metadata to an image to enhance/override SaveImage node metadata.
    This node processes the image and adds metadata that will be preserved when SaveImage saves it.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model": ("MODEL",),
                "positive_conditioning": ("CONDITIONING",),
                "negative_conditioning": ("CONDITIONING",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "scheduler": ("STRING", {"default": ""}),
                "sampler_name": ("STRING", {"default": ""}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "model_name": ("STRING", {"default": ""}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
                "width": ("INT", {"default": 512, "min": 1}),
                "height": ("INT", {"default": 512, "min": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_metadata"
    CATEGORY = "image/manual_metadata"
    
    def enhance_metadata(self, image, model=None, positive_conditioning=None, negative_conditioning=None, 
                        positive_prompt="", negative_prompt="", seed=-1, scheduler="", sampler_name="", 
                        steps=20, model_name="", cfg_scale=7.0, width=512, height=512):
        """
        Add metadata to the image. The metadata will be embedded in the image
        using PNG metadata format, which should be preserved when SaveImage saves it.
        Note: This works by embedding metadata in the image data itself.
        """
        import torch
        import numpy as np
        import tempfile
        
        # Process all images in the batch
        images_out = []
        for img_idx in range(len(image)):
            # Get single image from batch
            i = image[img_idx]
            
            # Convert to numpy and then to PIL
            i_np = 255. * i.cpu().numpy()
            img = Image.fromarray(np.clip(i_np, 0, 255).astype(np.uint8))
            
            # Create metadata dictionary
            metadata = {}
            
            # Extract prompts from CONDITIONING if provided, otherwise use string inputs
            final_positive_prompt = positive_prompt
            if positive_conditioning is not None:
                extracted_positive = get_prompt_from_conditioning(positive_conditioning)
                if extracted_positive:
                    final_positive_prompt = extracted_positive
            
            final_negative_prompt = negative_prompt
            if negative_conditioning is not None:
                extracted_negative = get_prompt_from_conditioning(negative_conditioning)
                if extracted_negative:
                    final_negative_prompt = extracted_negative
            
            if final_positive_prompt:
                metadata["positive_prompt"] = final_positive_prompt
                metadata["prompt"] = final_positive_prompt
            
            if final_negative_prompt:
                metadata["negative_prompt"] = final_negative_prompt
            
            if seed != -1:
                metadata["seed"] = str(seed)
            
            if scheduler:
                metadata["scheduler"] = scheduler
            
            if sampler_name:
                metadata["sampler"] = sampler_name
                metadata["sampler_name"] = sampler_name
            
            if steps:
                metadata["steps"] = str(steps)
            
            # Extract model name from MODEL input if provided, otherwise use string input
            final_model_name = model_name
            if model is not None:
                extracted_name = get_model_name(model)
                if extracted_name:
                    final_model_name = extracted_name
            
            if final_model_name:
                metadata["model"] = final_model_name
                metadata["model_name"] = final_model_name
            
            if cfg_scale:
                metadata["cfg_scale"] = str(cfg_scale)
            
            if width:
                metadata["width"] = str(width)
            
            if height:
                metadata["height"] = str(height)
            
            # Create workflow parameters dictionary for ComfyUI compatibility
            workflow_params = {}
            if final_positive_prompt:
                workflow_params["positive"] = final_positive_prompt
            if final_negative_prompt:
                workflow_params["negative"] = final_negative_prompt
            if seed != -1:
                workflow_params["seed"] = seed
            if scheduler:
                workflow_params["scheduler"] = scheduler
            if steps:
                workflow_params["steps"] = steps
            if final_model_name:
                workflow_params["model"] = final_model_name
            if cfg_scale:
                workflow_params["cfg"] = cfg_scale
            if width:
                workflow_params["width"] = width
            if height:
                workflow_params["height"] = height
            if sampler_name:
                workflow_params["sampler_name"] = sampler_name
            
            # Embed metadata using PngInfo
            pnginfo = PngImagePlugin.PngInfo()
            for key, value in metadata.items():
                pnginfo.add_text(key, str(value))
            
            # Store workflow parameters as JSON for ComfyUI
            if workflow_params:
                pnginfo.add_text("workflow", json.dumps(workflow_params))
            
            # Save to temporary location with metadata, then reload
            # This embeds the metadata in the PNG format
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(temp_file.name, pnginfo=pnginfo)
            
            # Reload image (now with embedded metadata)
            img_with_meta = Image.open(temp_file.name)
            
            # Convert back to tensor
            img_array = np.array(img_with_meta).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            images_out.append(img_tensor)
            
            # Clean up temp file
            os.unlink(temp_file.name)
        
        # Stack images back into batch
        result = torch.cat(images_out, dim=0)
        
        return (result,)


class ManualSaveImage:
    """
    Drop-in replacement for SaveImage with additional metadata inputs.
    This node saves images with manually specified metadata.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "scheduler": ("STRING", {"default": ""}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "model_name": ("STRING", {"default": ""}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
                "width": ("INT", {"default": 512, "min": 1}),
                "height": ("INT", {"default": 512, "min": 1}),
                "sampler_name": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/manual_metadata"
    
    def save_images(self, images, filename_prefix="ComfyUI", model=None, positive_conditioning=None, 
                   negative_conditioning=None, positive_prompt="", negative_prompt="", seed=-1, 
                   scheduler="", sampler_name="", steps=20, cfg_scale=7.0, width=512, height=512):
        """
        Save images with manually specified metadata.
        """
        import torch
        import numpy as np
        import datetime
        
        # Get output directory
        output_dir = folder_paths.get_output_directory()
        if self.type == "input":
            output_dir = folder_paths.get_input_directory()
        
        # Create filename
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )
        
        results = list()
        
        # Process each image in the batch
        for image in images:
            # Convert tensor to PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Create metadata
            metadata = PngImagePlugin.PngInfo()
            
            # Extract prompts from CONDITIONING if provided, otherwise use string inputs
            final_positive_prompt = positive_prompt
            if positive_conditioning is not None:
                extracted_positive = get_prompt_from_conditioning(positive_conditioning)
                if extracted_positive:
                    final_positive_prompt = extracted_positive
            
            final_negative_prompt = negative_prompt
            if negative_conditioning is not None:
                extracted_negative = get_prompt_from_conditioning(negative_conditioning)
                if extracted_negative:
                    final_negative_prompt = extracted_negative
            
            # Add standard metadata fields
            if final_positive_prompt:
                metadata.add_text("prompt", final_positive_prompt)
                metadata.add_text("positive_prompt", final_positive_prompt)
            
            if final_negative_prompt:
                metadata.add_text("negative_prompt", final_negative_prompt)
            
            if seed != -1:
                metadata.add_text("seed", str(seed))
            
            if scheduler:
                metadata.add_text("scheduler", scheduler)
            
            if sampler_name:
                metadata.add_text("sampler", sampler_name)
                metadata.add_text("sampler_name", sampler_name)
            
            if steps:
                metadata.add_text("steps", str(steps))
            
            # Extract model name from MODEL input if provided
            final_model_name = ""
            if model is not None:
                extracted_name = get_model_name(model)
                if extracted_name:
                    final_model_name = extracted_name
            
            if final_model_name:
                metadata.add_text("model", final_model_name)
                metadata.add_text("model_name", final_model_name)
            
            if cfg_scale:
                metadata.add_text("cfg_scale", str(cfg_scale))
            
            if width:
                metadata.add_text("width", str(width))
            
            if height:
                metadata.add_text("height", str(height))
            
            # Add workflow parameters for ComfyUI compatibility
            workflow_params = {}
            if final_positive_prompt:
                workflow_params["positive"] = final_positive_prompt
            if final_negative_prompt:
                workflow_params["negative"] = final_negative_prompt
            if seed != -1:
                workflow_params["seed"] = seed
            if scheduler:
                workflow_params["scheduler"] = scheduler
            if steps:
                workflow_params["steps"] = steps
            if final_model_name:
                workflow_params["model"] = final_model_name
            if cfg_scale:
                workflow_params["cfg"] = cfg_scale
            if width:
                workflow_params["width"] = width
            if height:
                workflow_params["height"] = height
            if sampler_name:
                workflow_params["sampler_name"] = sampler_name
            
            if workflow_params:
                metadata.add_text("workflow", json.dumps(workflow_params))
            
            # Add timestamp
            metadata.add_text("date", datetime.datetime.now().isoformat())
            
            # Generate filename with counter
            file = f"{filename}_{counter:05}_.png"
            file_path = os.path.join(full_output_folder, file)
            
            # Save image with metadata
            img.save(file_path, pnginfo=metadata, optimize=True)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            
            counter += 1
        
        return {"ui": {"images": results}}

