"""
ComfyUI Manual Metadata Plugin
Allows manual input and override of image metadata including prompts, seed, scheduler, steps, model name, etc.
"""

from .nodes import ManualMetadataEnhancer, ManualSaveImage

NODE_CLASS_MAPPINGS = {
    "ManualMetadataEnhancer": ManualMetadataEnhancer,
    "ManualSaveImage": ManualSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ManualMetadataEnhancer": "Manual Metadata Enhancer",
    "ManualSaveImage": "Manual Save Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

