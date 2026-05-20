import time 
import gc
from tqdm import tqdm
import random
import re
import torch
import psycopg2
from typing import Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data_collection.logging_config import logger 


class ModelDriver:
    """
    A utility class for inferencing causal language models on financial reports.
    Provides functionality for scoring, token probability inspection, and text generation.
    """

    default_verbolizer = {
        "positive": [
            "buy", "invest", "purchase", "Invest", "buying", "stay",
            "proceed", "recommend", "Hold", "retain", "increase",
            "maintain", "acquire"
        ],
        "negative": [
            "sell", "avoid", "caut", "carefully", "closely", "caution",
            "analyze", "minimize", "Avoid", "decrease", "Wait",
            "investigate", "sold", "decline", "Monitor", "assess",
            "sale", "remove", "seriously"
        ],
    }

    def __init__(self, model_name: str, model: AutoModelForCausalLM,
                 prompt: Optional[str] = None,
                 verbolizer: Optional[dict] = None):
        """
        Initializes the ModelTester with model, tokenizer, default prompt, and verbolizer.

        Args:
            model_name (str): HuggingFace model name for tokenizer loading.
            model (AutoModelForCausalLM): Pre-initialized transformer model.
            prompt (str, optional): Default prompt to append to inputs.
            verbolizer (dict, optional): Custom word lists for scoring.
        """
        self.model_name = model_name
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = next(model.parameters()).device.type

        self.default_prompt = prompt or "Based on this financial report my investment advice is to"

        self.verbolizers = [self.default_verbolizer]
        if verbolizer:
            self.verbolizers.append(verbolizer)

        for ind, base_words in enumerate(self.verbolizers):
            valid_verbolizer = self.create_verbolizer(base_words)
            self.verbolizers[ind] = valid_verbolizer

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
        Cleans raw financial reports while preserving table formatting.

        Args:
            report (str): Raw report string.

        Returns:
            str: Cleaned and formatted report string.
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

    def split_and_tokenize_report(self, report: str, max_tokens: int = 7989, overlap_ratio: float = 0.2) -> dict:
        """
        Splits a cleaned report into overlapping tokenized segments.

        Args:
            report (str): Cleaned report text.
            max_tokens (int): Max tokens per segment.
            overlap_ratio (float): Overlap between segments.

        Returns:
            dict: Mapping of segment names to tokenized tensors.
        """
        cleaned_report = self.clean_report(report)
        tokens = self.tokenizer(cleaned_report, return_tensors="pt")['input_ids'].squeeze().to(self.device)
        prompt_tokens = self.tokenizer(self.default_prompt, return_tensors="pt")['input_ids'].squeeze().to(self.device)

        token_segments = {}
        start = 0
        segment_index = 1
        overlap_tokens = int(max_tokens * overlap_ratio)

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            segment_tokens = torch.cat((tokens[start:end], prompt_tokens), dim=0).to(self.device)
            logger.debug(f"Segment length: {segment_tokens.shape[0]}")
            token_segments[f"Segment_{segment_index}"] = segment_tokens
            start += max_tokens - overlap_tokens
            segment_index += 1

        return token_segments

    def fast_inference(self, tokens: torch.Tensor) -> dict:
        """
        Runs inference and returns next-token probabilities for valid tokens.

        Args:
            tokens (torch.Tensor): Tokenized input segment.

        Returns:
            dict: Mapping of token strings to probability values.
        """
        try:
            logger.debug(f"Input shape: {tokens.shape}")

            inputs = {"input_ids": tokens.unsqueeze(0).to(self.device)}
           
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze()
      
            self.masked_logits.fill_(-float("inf"))
            self.masked_logits[self.valid_mask] = logits.to(self.masked_logits.dtype)[self.valid_mask]

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
        Generates a continuation based on given text and a prompt.

        Args:
            text (str): Input text (optional).
            custom_prompt (str): Custom prompt (optional).
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            str: Generated continuation text.
        """
        if not text:
            text = self.get_random_reports()[0]
        prompt = custom_prompt or self.default_prompt
        text += prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def get_score(self, token_prob_dict: dict,) -> list[float]:
        """
        Computes sentiment score from token probability dictionary.

        Args:
            token_prob_dict (dict): Token-probability pairs.

        Returns:
            float: Sentiment score (P(positive) - P(negative)).
        """
        scores = []

        for v in self.verbolizers:
            positive_prob = sum(token_prob_dict.get(word, 0) for word in v["positive"])
            negative_prob = sum(token_prob_dict.get(word, 0) for word in v["negative"])
            score = positive_prob - negative_prob
            scores.append(score)

        return scores

    def infer_report_type(
        self,
        text: Optional[str] = None,
        context_max_tokens: int = 8000,
        prompt: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Infers report type probabilities (10-K vs 10-Q) from a report snippet.

        Logic:
        - Uses one shared context window (~8000 tokens total budget with prompt),
        - Appends task prompt,
        - Runs next-token inference,
        - Aggregates verbalizer-token probabilities per class,
        - Returns normalized class probabilities.

        Args:
            text (str, optional): Report text. If None, random report is used.
            context_max_tokens (int): Token budget for context + prompt.
            prompt (str, optional): Custom task prompt.

        Returns:
            dict[str, float]: {"10-K": p_k, "10-Q": p_q}
        """
        if not text:
            text = self.get_random_reports()[0]
        #"You just saw a part of financial report. Question: This SEC filing is annual or quarterly? Answer with one word: annual or quarterly. Based on this part of report my answer is"
        task_prompt = prompt or (
            "You just saw a part of financial report. \n"
            "Question: This SEC filing is annual or quarterly? Answer with one word: annual or quarterly. \n"
            "Based on this part of report my answer is"
        )

        clean_text = self.clean_report(text)

        prompt_tokens = self.tokenizer(task_prompt, return_tensors="pt")["input_ids"].squeeze()
        if prompt_tokens.dim() == 0:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        max_report_tokens = max(1, context_max_tokens - prompt_tokens.shape[0])
        report_tokens = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_report_tokens,
        )["input_ids"].squeeze()
        if report_tokens.dim() == 0:
            report_tokens = report_tokens.unsqueeze(0)

        tokens = torch.cat((report_tokens, prompt_tokens), dim=0).to(self.device)
        token_prob_dict = self.fast_inference(tokens)
        sorted_items = sorted(
            token_prob_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(sorted_items[:20])
        verbolizer = {
            "10-K": ["A"],
            "10-Q": ["B"],
        }

        scores: dict[str, float] = {}
        for label, words in verbolizer.items():
            expanded_words = set()
            for word in words:
                lower = word.lower()
                expanded_words.add(lower)
                expanded_words.add(lower.capitalize())

            scores[label] = sum(token_prob_dict.get(word, 0.0) for word in expanded_words)
        
    

        total_score = sum(scores.values())
        if total_score <= 0:
            logger.warning("Report type scores sum to 0. Returning zeros.")
            return {label: 0.0 for label in scores}

        return {label: score / total_score for label, score in scores.items()}

    def compute_sample_scores(self, text: Optional[str] = None) -> tuple[list[list[float]], float]:
        """
        Computes scores for each tokenized segment of a report.

        Args:
            text (str): Report text (optional).

        Returns:
            list[float]: Scores for each segment.
        """
        if not text:
            text = self.get_random_reports()[0]

        clean_text = self.clean_report(text)
        tokenized_segments = self.split_and_tokenize_report(clean_text)

        self.sample_scores: list[list[float]] = []

        logger.debug(f"There are {len(tokenized_segments)} segments of {len(tokenized_segments['Segment_1'])} len")
        
        inference_durations: list[float] = []

        for segment_name, tokens in tqdm(
            tokenized_segments.items(), 
            desc="Processing Segments", 
            unit="segment",
            position=0,
            ):
            logger.debug(f"Processing {segment_name}...")

            start = time.perf_counter()
            tokens = tokens.to(self.device)
            token_prob_dict = self.fast_inference(tokens)
            end = time.perf_counter()

            inference_time = end - start
            inference_durations.append(inference_time)

            sample_score: list[float] = self.get_score(token_prob_dict,)
            logger.debug(sample_score)

            self.sample_scores.append(sample_score)

            del tokens
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
        
        inference_durations.pop()
        self.avg_inefence_duration: float = sum(inference_durations) / len(inference_durations)

        return self.sample_scores, self.avg_inefence_duration

    def see_pmf(self, text: str | None, top_k_tokens: int = 20):
        """
        Displays the top-k most probable tokens based on model output.

        Args:
            text (str): Optional input report.
            top_k_tokens (int): Number of top tokens to display.
        """
        if not text:
            text = self.get_random_reports()[0]

        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze().to(self.device)

        token_prob_dict = self.fast_inference(tokens)

        sorted_token_probs = {k: v for k, v in sorted(token_prob_dict.items(), key=lambda item: item[1], reverse=True)}
        print("\n🔹 **Next Token Prediction Probabilities (Only Meaningful Words):**")
        for token, prob in list(sorted_token_probs.items())[:top_k_tokens]:  
            print(f"{token:<10} | Probability: {prob:.4f}")

    def get_random_reports(self, n: int = 1, seed: int = 42) -> list[str]:
        """
        Fetches n random financial reports from a local PostgreSQL DB.

        Args:
            n (int): Number of reports.
            seed (int): Seed for sampling.

        Returns:
            list[str]: Fetched raw report strings.
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
                cur.execute("SELECT id FROM reports WHERE raw_text IS NOT NULL;")
                valid_ids = [row[0] for row in cur.fetchall()]

                if not valid_ids:
                    raise ValueError("No reports with raw_text found in the database.")

                if len(valid_ids) < n:
                    print(f"⚠️ Only {len(valid_ids)} reports available. Returning all.")
                    n = len(valid_ids)

                random.seed(seed)
                sampled_ids = random.sample(valid_ids, n)

                cur.execute(
                    "SELECT raw_text FROM reports WHERE id = ANY(%s);",
                    (sampled_ids,)
                )
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def create_verbolizer(base_words: dict[str: str]) -> dict[str: str]:
        """
        Converts word list dict into lowercase+capitalized variants.

        Args:
            base_words (dict): Words under "positive" and "negative" keys.

        Returns:
            dict: Expanded keyword list for scoring.
        """
        positive_words = list(map(str.lower, base_words["positive"]))
        negative_words = list(map(str.lower, base_words["negative"]))
        return {
            "positive": [w.capitalize() for w in positive_words] + positive_words,
            "negative": [w.capitalize() for w in negative_words] + negative_words,
        }
