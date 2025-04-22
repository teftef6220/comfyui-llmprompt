# from .text_input_node import NODE_CLASS_MAPPINGS

NODE_CLASS_MAPPINGS = {}

from . import text_input_node, llm_nodes, process_image_nodes

for module in [text_input_node, llm_nodes, process_image_nodes]:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
