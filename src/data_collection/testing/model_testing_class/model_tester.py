import gc
import random
import re
import torch
import psycopg2
from typing import Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelTester:

    def __init__(self, model_name: str, model: AutoModelForCausalLM, device: str = "auto",
                 prompt: Optional[str] = None,
                 verbolizer: Optional[dict] = None):
        """
        Initializes the ModelTester with model name, model object, device specification,
        default prompt, and default verbolizer.

        Args:
            model_name (str): Name of the HuggingFace model.
            model (AutoModelForCausalLM): Preloaded model object.
            device (str): Target device for model and tensors ("cuda", "cpu", or "auto").
            default_prompt (str, optional): Default prompt to append to inputs.
            verbolizer (dict, optional): Dictionary of positive/negative words for scoring.
        """
        self.model_name = model_name
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device_map = "auto"
        else:
            self.device = device
            self.device_map = {"": device}

        self.default_prompt = prompt or "Based on this financial report my investment advice is to"
        self.verbolizer = self.create_verbolizer(verbolizer) if verbolizer else self.create_verbolizer(
            positive_words=[
                "buy", "invest", "purchase", "Invest", "buying", "stay",
                "proceed", "recommend", "Hold", "retain", "increase",
                "maintain", "acquire"
            ],
            negative_words=[
                "sell", "avoid", "caut", "carefully", "closely", "caution",
                "analyze", "minimize", "Avoid", "decrease", "Wait",
                "investigate", "sold", "decline", "Monitor", "assess",
                "sale", "remove", "seriously"
            ]
        )

        # Initialize valid token mask
        vocab_size = len(self.tokenizer.get_vocab())
        all_tokens = torch.arange(vocab_size, device=self.device)
        decoded_tokens = self.tokenizer.batch_decode(all_tokens.unsqueeze(1))

        self.valid_mask = torch.tensor([
            token.isalpha() and len(token) > 1 for token in decoded_tokens
        ], dtype=torch.bool, device=self.device).clone()

        self.allowed_tokens = torch.masked_select(all_tokens, self.valid_mask)
        self.allowed_token_texts = self.tokenizer.batch_decode(self.allowed_tokens.tolist())
        self.masked_logits = torch.empty(vocab_size, dtype=self.model.config.torch_dtype or torch.float32, device=self.device)

    def clean_report(self, report: str) -> str:
        """
        Cleans the input report by removing excess whitespace while preserving table structure.

        Args:
            report (str): Raw report text.

        Returns:
            str: Cleaned report text.
        """
        lines = report.split("\n")
        cleaned_lines = []
        for line in lines:
            cleaned_line = re.sub(r"\s+", " ", line).strip()
            if "|" in line:
                cleaned_lines.append(line)
            elif cleaned_line:
                cleaned_lines.append(cleaned_line)
        return "\n".join(cleaned_lines)

    def split_and_tokenize_report(self, report: str, max_tokens: int = 3800, overlap_ratio: float = 0.2) -> dict:
        """
        Splits a cleaned report into overlapping tokenized segments suitable for model inference.

        Args:
            report (str): Cleaned report text.
            max_tokens (int): Max tokens per segment.
            overlap_ratio (float): Overlap ratio for segment continuation.

        Returns:
            dict: Dictionary mapping segment labels to tokenized tensors.
        """
        cleaned_report = self.clean_report(report)

        tokens = self.tokenizer(cleaned_report, return_tensors="pt")['input_ids'].squeeze().to("cpu")
        prompt_tokens = self.tokenizer(self.default_prompt, return_tensors="pt")['input_ids'].squeeze().to("cpu")

        token_segments = {}
        start = 0
        segment_index = 1
        overlap_tokens = int(max_tokens * overlap_ratio)

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            segment_tokens = torch.cat((tokens[start:end], prompt_tokens), dim=0).to("cpu")
            token_segments[f"Segment_{segment_index}"] = segment_tokens
            start += max_tokens - overlap_tokens
            segment_index += 1

        return token_segments

    def fast_inference(self, tokens: torch.Tensor) -> dict:
        """
        Runs a forward pass to compute probability distribution over allowed next tokens.

        Args:
            tokens (torch.Tensor): Tokenized input segment.

        Returns:
            dict: Dictionary mapping allowed token strings to their probabilities.
        """
        try:
            inputs = {"input_ids": tokens.unsqueeze(0).to(self.device)}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze()

            self.masked_logits.fill_(-float("inf"))
            self.masked_logits[self.valid_mask] = logits[self.valid_mask]

            probs = torch.nn.functional.softmax(self.masked_logits, dim=-1)
            allowed_probs = torch.masked_select(probs, self.valid_mask)

            return dict(zip(self.allowed_token_texts, allowed_probs.tolist()))

        finally:
            for name in ["tokens", "inputs", "outputs", "logits", "probs", "allowed_probs"]:
                if name in locals():
                    del locals()[name]
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

    def generate_text(self, text: Optional[str] = None, custom_prompt: Optional[str] = None, max_new_tokens: int = 64) -> str:
        """
        Generates text completion from a given or random report.

        Args:
            text (str, optional): Custom report text. If None, randomly selected.
            custom_prompt (str, optional): Prompt to append. If None, uses default.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            str: Generated text continuation.
        """
        if not text:
            text = self.get_random_report()
        prompt = custom_prompt or self.default_prompt
        text += prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id  # prevent pad warning
            )
        generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def get_score(self, token_prob_dict: dict,) -> float:
        """
        Computes sentiment score using a given or default verbolizer.

        Args:
            token_prob_dict (dict): Token probability distribution.
            verbolizer (dict, optional): Word list dict with 'positive' and 'negative'.

        Returns:
            float: Sentiment score = P(positive) - P(negative).
        """
        v = self.verbolizer
        positive_prob = sum(token_prob_dict.get(word, 0) for word in v["positive"])
        negative_prob = sum(token_prob_dict.get(word, 0) for word in v["negative"])
        return positive_prob - negative_prob

    def compute_sample_scores(self, text: str | None) -> None:
        """
        Computes and prints sentiment scores for a randomly selected report.

        Args:
            verbolizer (dict, optional): Custom word scoring dictionary.
        """
        if not text:
            text = self.get_random_report()

        clean_text = self.clean_report(text)
        tokenized_segments = self.split_and_tokenize_report(clean_text)

        for segment_name, tokens in tokenized_segments.items():
            print(f"Processing {segment_name}...")
            tokens = tokens.to(self.device)
            token_prob_dict = self.fast_inference(tokens)
            print(self.get_score(token_prob_dict,))
            del tokens
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

    def see_pmf(self, text: str | None, top_k_tokens: int = 20):
        """
        Displays the top-k token probabilities from the model's next-token distribution.

        This method runs inference on a given text (or a randomly fetched financial report if `text` is None),
        computes the probability distribution over all allowed tokens (filtered to exclude punctuation, digits, etc.),
        and prints the top-k tokens with their associated probabilities.

        Args:
            text (str | None): Optional custom input text. If None, a random financial report will be fetched.
            top_k_tokens (int): Number of top probable tokens to display from the model's output.

        Returns:
            None. Prints the results directly.
        """
        if not text:
            text = self.get_random_report()

        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze().to(self.device)

        token_prob_dict = self.fast_inference(tokens)
    
        sorted_token_probs = {k: v for k, v in sorted(token_prob_dict.items(), key=lambda item: item[1], reverse=True)}
        print("\n🔹 **Next Token Prediction Probabilities (Only Meaningful Words):**")
        for token, prob in list(sorted_token_probs.items())[:top_k_tokens]:  
            print(f"{token:<10} | Probability: {prob:.4f}")

    def get_random_report(self) -> str:
        """
        Retrieves a random financial report from the PostgreSQL database.

        Returns:
            str: Raw report text.
        """
        conn = psycopg2.connect(
            dbname="reports_db",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM reports;")
                total_reports = cur.fetchone()[0]
                if total_reports == 0:
                    raise ValueError("No reports found in the database.")

                offset = random.randint(0, total_reports - 1)
                cur.execute("SELECT raw_text FROM reports OFFSET %s LIMIT 1;", (offset,))
                return cur.fetchone()[0]
        finally:
            conn.close()

    @staticmethod
    def create_verbolizer(positive_words: list[str], negative_words: list[str]) -> dict:
        """
        Constructs a dictionary of lower- and capitalized sentiment keywords.

        Args:
            positive_words (list[str]): List of positive keywords.
            negative_words (list[str]): List of negative keywords.

        Returns:
            dict: A dict with 'positive' and 'negative' mapped to keyword variants.
        """
        positive_words = list(map(str.lower, positive_words))
        negative_words = list(map(str.lower, negative_words))
        return {
            "positive": [w.capitalize() for w in positive_words] + positive_words,
            "negative": [w.capitalize() for w in negative_words] + negative_words,
        }



