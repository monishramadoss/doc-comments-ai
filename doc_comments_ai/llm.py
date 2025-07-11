from huggingface_hub import InferenceClient

class LLM:
    def __init__(
        self,
        model: str,
    ):
        self.client = InferenceClient(model=model)

        self.template = (
            "Add a detailed doc comment to the following {language} method:\n{code}\n"
            "The doc comment should describe what the method does. "
            "{inline_comments} "
            "Return the method implementaion with the doc comment as a single markdown code block. "
            "Don't include any explanations {haskell_missing_signature}in your response."
        )

    def generate_doc_comment(self, language, code, inline=False):
        """
        Generates a doc comment for the given method
        """

        if inline:
            inline_comments = (
                "Add inline comments to the method body where it makes sense."
            )
        else:
            inline_comments = ""

        if language == "haskell":
            haskell_missing_signature = "and missing type signatures "
        else:
            haskell_missing_signature = ""

        prompt = self.template.format(
            language=language,
            code=code,
            inline_comments=inline_comments,
            haskell_missing_signature=haskell_missing_signature,
        )

        documented_code = self.client.text_generation(prompt, max_new_tokens=1024)

        return documented_code
