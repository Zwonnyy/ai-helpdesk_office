from ollama import chat


MODEL_NAME = "qwen3:4b"


class LLMService:

    def __init__(self):
        self.model_name = MODEL_NAME

    def build_prompt(
        self,
        subject: str,
        body: str,
        predictions: dict,
        similar_tickets: list,
    ) -> str:

        references = []

        for index, ticket in enumerate(
            similar_tickets,
            start=1,
        ):
            references.append(
                f"""
[REFERENCE {index}]

Similarity:
{ticket["score"]:.4f}

Subject:
{ticket["subject"]}

Body:
{ticket["body"]}

Previous Answer:
{ticket["answer"]}

Type:
{ticket["type"]}

Queue:
{ticket["queue"]}

Priority:
{ticket["priority"]}
"""
            )

        reference_text = "\n".join(
            references
        )

        prompt = f"""
You are an IT Helpdesk support assistant.

Your job is to draft a response to a new customer support ticket.

IMPORTANT RULES:

1. Use the previous support tickets only as reference.
2. Do not claim facts that are not supported by the new ticket or references.
3. Do not invent troubleshooting results.
4. If information is insufficient, ask the user for the missing information.
5. Do not mention that AI generated the answer.
6. Answer in the same language as the user's ticket.
7. Keep the answer professional and concise.
8. Do not blindly copy previous answers.
9. Adapt the response to the new ticket.
10. Do not expose internal classification labels to the customer.

============================================================
NEW TICKET
============================================================

Subject:
{subject}

Body:
{body}

============================================================
AI CLASSIFICATION
============================================================

Type:
{predictions["type"]["label"]}

Type Confidence:
{predictions["type"]["confidence"]:.4f}

Queue:
{predictions["queue"]["label"]}

Queue Confidence:
{predictions["queue"]["confidence"]:.4f}

Priority:
{predictions["priority"]["label"]}

Priority Confidence:
{predictions["priority"]["confidence"]:.4f}

============================================================
SIMILAR PREVIOUS TICKETS
============================================================

{reference_text}

============================================================
TASK
============================================================

Write a customer-facing support response draft.

Only output the response that should be sent to the customer.
"""

        return prompt

    def generate_answer(
        self,
        subject: str,
        body: str,
        predictions: dict,
        similar_tickets: list,
    ) -> str:

        prompt = self.build_prompt(
            subject=subject,
            body=body,
            predictions=predictions,
            similar_tickets=similar_tickets,
        )

        response = chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return (
            response
            .message
            .content
            .strip()
        )
