# text_input_node.py

class PromptInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "This is a test prompt.",
                    "multiline": True
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "send"
    CATEGORY = "MyPromptTest"

    def send(self, prompt):
        return (prompt,)


class AddPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_1": ("STRING", {
                    "default": "first prompt.",
                    "multiline": True
                }),
                "prompt_2": ("STRING", {
                    "default": "last prompt.",
                    "multiline": True
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "add"
    CATEGORY = "MyPromptTest"

    def add(self, prompt_1,prompt_2):
        return ((prompt_1 +", "+ prompt_2),)
        


NODE_CLASS_MAPPINGS = {
    "PromptInput": PromptInput,
    "AddPrompt":AddPrompt,
}
