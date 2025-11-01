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


def get_sampler_info(sampler):
    """Extract sampler, scheduler, and steps from SAMPLER type object."""
    if sampler is None:
        return None, None, None
    
    sampler_name = None
    scheduler_name = None
    steps = None
    
    # Try common attributes
    if hasattr(sampler, 'sampler'):
        sampler_name = str(sampler.sampler) if sampler.sampler else None
    if hasattr(sampler, 'scheduler'):
        scheduler_name = str(sampler.scheduler) if sampler.scheduler else None
    if hasattr(sampler, 'steps'):
        steps = int(sampler.steps) if sampler.steps else None
    
    # Try alternative attribute names
    if not sampler_name:
        for attr in ['sampler_name', 'name', 'method']:
            if hasattr(sampler, attr):
                sampler_name = str(getattr(sampler, attr))
                break
    
    if not scheduler_name:
        for attr in ['scheduler_name', 'scheduling_method']:
            if hasattr(sampler, attr):
                scheduler_name = str(getattr(sampler, attr))
                break
    
    # Try accessing inner objects
    if hasattr(sampler, 'sampler_name'):
        sampler_name = str(sampler.sampler_name) if sampler.sampler_name else None
    
    # Get class name as fallback
    if not sampler_name:
        sampler_name = sampler.__class__.__name__ if hasattr(sampler, '__class__') else None
    
    return sampler_name, scheduler_name, steps


def get_image_dimensions(image):
    """Extract width and height from IMAGE tensor."""
    if image is None:
        return None, None
    # IMAGE tensor shape is typically (batch, height, width, channels)
    if len(image.shape) >= 3:
        height = int(image.shape[1]) if image.shape[1] > 0 else None
        width = int(image.shape[2]) if image.shape[2] > 0 else None
        return width, height
    return None, None


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
                "sampler": ("SAMPLER",),
                "positive_conditioning": ("CONDITIONING",),
                "negative_conditioning": ("CONDITIONING",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Leave blank if using CONDITIONING inputs"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Leave blank if using CONDITIONING inputs"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "scheduler": ("STRING", {"default": "", "tooltip": "Automatically detected from SAMPLER input if left blank"}),
                "sampler_name": ("STRING", {"default": "", "tooltip": "Automatically detected from SAMPLER input if left blank"}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Automatically detected from SAMPLER input if set to -1"}),
                "model_name": ("STRING", {"default": ""}),
                "cfg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0}),
                "width": ("INT", {"default": -1, "min": -1, "tooltip": "Automatically detected from image dimensions if set to -1"}),
                "height": ("INT", {"default": -1, "min": -1, "tooltip": "Automatically detected from image dimensions if set to -1"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_metadata"
    CATEGORY = "image/manual_metadata"
    
    def enhance_metadata(self, image, model=None, sampler=None, positive_conditioning=None, negative_conditioning=None, 
                        positive_prompt="", negative_prompt="", seed=-1, scheduler="", sampler_name="", 
                        steps=-1, model_name="", cfg_scale=0.0, width=-1, height=-1):
        """
        Add metadata to the image. The metadata will be embedded in the image
        using PNG metadata format, which should be preserved when SaveImage saves it.
        Note: This works by embedding metadata in the image data itself.
        """
        import torch
        import numpy as np
        import tempfile
        
        # Extract sampler info if provided
        auto_sampler_name, auto_scheduler, auto_steps = get_sampler_info(sampler)
        
        # Auto-detect width/height from first image
        auto_width, auto_height = get_image_dimensions(image[0] if len(image) > 0 else None)
        
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
            
            # Use manual scheduler if provided, otherwise use auto-detected
            final_scheduler = scheduler if scheduler else (auto_scheduler if auto_scheduler else "")
            if final_scheduler:
                metadata["scheduler"] = final_scheduler
            
            # Use manual sampler_name if provided, otherwise use auto-detected
            final_sampler_name = sampler_name if sampler_name else (auto_sampler_name if auto_sampler_name else "")
            if final_sampler_name:
                metadata["sampler"] = final_sampler_name
                metadata["sampler_name"] = final_sampler_name
            
            # Use manual steps if provided (> -1), otherwise use auto-detected
            final_steps = steps if steps >= 0 else (auto_steps if auto_steps else -1)
            if final_steps and final_steps >= 0:
                metadata["steps"] = str(final_steps)
            
            # Extract model name from MODEL input if provided, otherwise use string input
            final_model_name = model_name
            if model is not None:
                extracted_name = get_model_name(model)
                if extracted_name:
                    final_model_name = extracted_name
            
            if final_model_name:
                metadata["model"] = final_model_name
                metadata["model_name"] = final_model_name
            
            # Use manual cfg_scale if provided (> 0)
            if cfg_scale and cfg_scale > 0:
                metadata["cfg_scale"] = str(cfg_scale)
            
            # Use manual width if provided (>= 0), otherwise use auto-detected
            final_width = width if width >= 0 else (auto_width if auto_width else -1)
            if final_width and final_width >= 0:
                metadata["width"] = str(final_width)
            
            # Use manual height if provided (>= 0), otherwise use auto-detected
            final_height = height if height >= 0 else (auto_height if auto_height else -1)
            if final_height and final_height >= 0:
                metadata["height"] = str(final_height)
            
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
                "sampler": ("SAMPLER",),
                "positive_conditioning": ("CONDITIONING",),
                "negative_conditioning": ("CONDITIONING",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Leave blank if using CONDITIONING inputs"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Leave blank if using CONDITIONING inputs"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "scheduler": ("STRING", {"default": "", "tooltip": "Automatically detected from SAMPLER input if left blank"}),
                "sampler_name": ("STRING", {"default": "", "tooltip": "Automatically detected from SAMPLER input if left blank"}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Automatically detected from SAMPLER input if set to -1"}),
                "cfg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0}),
                "width": ("INT", {"default": -1, "min": -1, "tooltip": "Automatically detected from image dimensions if set to -1"}),
                "height": ("INT", {"default": -1, "min": -1, "tooltip": "Automatically detected from image dimensions if set to -1"}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/manual_metadata"
    
    def save_images(self, images, filename_prefix="ComfyUI", model=None, sampler=None, positive_conditioning=None, 
                   negative_conditioning=None, positive_prompt="", negative_prompt="", seed=-1, 
                   scheduler="", sampler_name="", steps=-1, cfg_scale=0.0, width=-1, height=-1):
        """
        Save images with manually specified metadata.
        """
        import torch
        import numpy as np
        import datetime
        
        # Extract sampler info if provided
        auto_sampler_name, auto_scheduler, auto_steps = get_sampler_info(sampler)
        
        # Auto-detect width/height from first image
        auto_width, auto_height = get_image_dimensions(images[0] if len(images) > 0 else None)
        
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
            
            # Use manual scheduler if provided, otherwise use auto-detected
            final_scheduler = scheduler if scheduler else (auto_scheduler if auto_scheduler else "")
            if final_scheduler:
                metadata.add_text("scheduler", final_scheduler)
            
            # Use manual sampler_name if provided, otherwise use auto-detected
            final_sampler_name = sampler_name if sampler_name else (auto_sampler_name if auto_sampler_name else "")
            if final_sampler_name:
                metadata.add_text("sampler", final_sampler_name)
                metadata.add_text("sampler_name", final_sampler_name)
            
            # Use manual steps if provided (>= 0), otherwise use auto-detected
            final_steps = steps if steps >= 0 else (auto_steps if auto_steps else -1)
            if final_steps and final_steps >= 0:
                metadata.add_text("steps", str(final_steps))
            
            # Extract model name from MODEL input if provided
            final_model_name = ""
            if model is not None:
                extracted_name = get_model_name(model)
                if extracted_name:
                    final_model_name = extracted_name
            
            if final_model_name:
                metadata.add_text("model", final_model_name)
                metadata.add_text("model_name", final_model_name)
            
            # Use manual cfg_scale if provided (> 0)
            if cfg_scale and cfg_scale > 0:
                metadata.add_text("cfg_scale", str(cfg_scale))
            
            # Use manual width if provided (>= 0), otherwise use auto-detected
            final_width = width if width >= 0 else (auto_width if auto_width else -1)
            if final_width and final_width >= 0:
                metadata.add_text("width", str(final_width))
            
            # Use manual height if provided (>= 0), otherwise use auto-detected
            final_height = height if height >= 0 else (auto_height if auto_height else -1)
            if final_height and final_height >= 0:
                metadata.add_text("height", str(final_height))
            
            # Add workflow parameters for ComfyUI compatibility
            workflow_params = {}
            if final_positive_prompt:
                workflow_params["positive"] = final_positive_prompt
            if final_negative_prompt:
                workflow_params["negative"] = final_negative_prompt
            if seed != -1:
                workflow_params["seed"] = seed
            if final_scheduler:
                workflow_params["scheduler"] = final_scheduler
            if final_steps and final_steps >= 0:
                workflow_params["steps"] = final_steps
            if final_model_name:
                workflow_params["model"] = final_model_name
            if cfg_scale and cfg_scale > 0:
                workflow_params["cfg"] = cfg_scale
            if final_width and final_width >= 0:
                workflow_params["width"] = final_width
            if final_height and final_height >= 0:
                workflow_params["height"] = final_height
            if final_sampler_name:
                workflow_params["sampler_name"] = final_sampler_name
            
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

