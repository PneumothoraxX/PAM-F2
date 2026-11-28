"""BFCL-style function-calling catalog, evaluation examples, and Tree Batching scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from src.core.prompts import DecisionExample, format_decision_prompt


@dataclass(frozen=True)
class BFCLToolSchema:
    """Schema definition for an executable API/function in the BFCL catalog."""

    tool_id: int
    name: str
    description: str
    parameters: dict[str, str]
    domain: str

    @property
    def prompt_label(self) -> str:
        params = ", ".join(self.parameters.keys())
        return f"{self.name}({params}): {self.description}"


@dataclass(frozen=True)
class BFCLQuerySample:
    """Single function-calling evaluation sample (clear or ambiguous/multi-constraint)."""

    query: str
    class_idx: int
    is_ambiguous: bool
    required_arguments: dict[str, str]
    domain: str


@dataclass(frozen=True)
class BFCLTreeScenario:
    """Shared context state paired with M concurrent decision questions for Tree Batching."""

    state_text: str
    questions: list[str]
    class_idx_list: list[int]


@dataclass(frozen=True)
class BFCLBenchmarkData:
    """Container for Phase 2 BFCL tools, evaluation samples, and tree-batch scenarios."""

    tools: list[BFCLToolSchema]
    samples: list[BFCLQuerySample]
    tree_scenarios: list[BFCLTreeScenario]

    @property
    def option_names(self) -> list[str]:
        return [t.name for t in self.tools]

    @property
    def option_Signature_labels(self) -> list[str]:
        return [t.prompt_label for t in self.tools]

    @property
    def num_options(self) -> int:
        return len(self.tools)


DEFAULT_BFCL_TOOLS: list[BFCLToolSchema] = [
    BFCLToolSchema(0, "get_account_balance", "Retrieve current checking or savings balance", {"account_id": "str", "currency": "str"}, "banking"),
    BFCLToolSchema(1, "transfer_funds", "Execute domestic wire or internal account transfer", {"from_acc": "str", "to_acc": "str", "amount": "float"}, "banking"),
    BFCLToolSchema(2, "lock_payment_card", "Temporarily freeze or permanently block a debit/credit card", {"card_id": "str", "reason": "str"}, "security"),
    BFCLToolSchema(3, "get_exchange_rate", "Fetch real-time FX conversion rate between two currencies", {"base_currency": "str", "target_currency": "str"}, "fx"),
    BFCLToolSchema(4, "dispute_card_charge", "Open a chargeback dispute for an unauthorized merchant transaction", {"tx_id": "str", "claim_reason": "str"}, "security"),
    BFCLToolSchema(5, "schedule_recurring_bill", "Configure automatic periodic payment for utilities or invoices", {"payee": "str", "amount": "float", "frequency": "str"}, "payments"),
    BFCLToolSchema(6, "query_transaction_history", "Search past account transactions filtered by date and category", {"account_id": "str", "start_date": "str", "end_date": "str"}, "banking"),
    BFCLToolSchema(7, "calculate_loan_amortization", "Compute monthly installment and total interest for a loan quote", {"principal": "float", "annual_rate": "float", "months": "int"}, "loans"),
    BFCLToolSchema(8, "create_virtual_card", "Issue a single-use or merchant-locked virtual card number", {"spending_limit": "float", "expires_in_days": "int"}, "cards"),
    BFCLToolSchema(9, "update_travel_notice", "Register international travel dates to prevent fraud blocks abroad", {"country_code": "str", "departure_date": "str", "return_date": "str"}, "security"),
    BFCLToolSchema(10, "export_tax_statement", "Generate annual PDF tax withholding and interest summary", {"tax_year": "int", "format": "str"}, "compliance"),
    BFCLToolSchema(11, "place_limit_stock_order", "Submit a limit buy or sell order on an equity ticker", {"ticker": "str", "shares": "int", "limit_price": "float"}, "investing"),
    BFCLToolSchema(12, "get_stock_market_quote", "Fetch latest bid/ask price and daily volume for a stock symbol", {"ticker": "str", "exchange": "str"}, "investing"),
    BFCLToolSchema(13, "cancel_active_subscription", "Revoke recurring merchant billing authorization", {"merchant_name": "str", "immediate": "bool"}, "payments"),
    BFCLToolSchema(14, "verify_identity_kyc", "Submit document verification status check for compliance review", {"user_id": "str", "document_type": "str"}, "compliance"),
    BFCLToolSchema(15, "set_budget_alert", "Create push notification threshold when category spend exceeds limit", {"category": "str", "monthly_limit": "float"}, "analytics"),
    BFCLToolSchema(16, "find_nearby_atms", "Locate fee-free ATMs within a radius of a postal code or city", {"location": "str", "radius_km": "float"}, "banking"),
    BFCLToolSchema(17, "convert_crypto_holding", "Swap between fiat balance and supported cryptocurrency assets", {"from_asset": "str", "to_asset": "str", "amount": "float"}, "crypto"),
    BFCLToolSchema(18, "split_shared_expense", "Send itemized peer-to-peer payment requests to multiple contacts", {"total_amount": "float", "participants": "list"}, "payments"),
    BFCLToolSchema(19, "redeem_cashback_rewards", "Apply accumulated loyalty points toward statement credit", {"points": "int", "destination_account": "str"}, "rewards"),
    BFCLToolSchema(20, "request_credit_limit_increase", "Apply for a higher credit line on an existing revolving card", {"card_id": "str", "requested_limit": "float", "annual_income": "float"}, "cards"),
    BFCLToolSchema(21, "register_wire_beneficiary", "Add and validate a new external IBAN/SWIFT recipient", {"recipient_name": "str", "iban": "str", "swift_bic": "str"}, "payments"),
    BFCLToolSchema(22, "estimate_wire_transfer_fee", "Calculate international remittance fees and delivery time", {"destination_country": "str", "amount": "float", "currency": "str"}, "fx"),
    BFCLToolSchema(23, "check_mortgage_preapproval", "Evaluate debt-to-income ratio for home mortgage eligibility", {"annual_income": "float", "down_payment": "float", "property_value": "float"}, "loans"),
]


_CLEAR_TEMPLATES: dict[int, list[str]] = {
    0: ["What is the available balance in my checking account {acc} in {curr}?", "Check how much money I have left in savings account {acc}."],
    1: ["Transfer ${amt} from account {acc} to account {acc2} right now.", "Send ${amt} immediately between my checking {acc} and savings {acc2}."],
    2: ["I lost my wallet at the airport, freeze my credit card {card} immediately!", "Block my debit card {card} right away because it was stolen."],
    3: ["What is the current exchange rate from {curr} to EUR today?", "Check the live FX conversion rate between USD and {curr}."],
    4: ["I see an unrecognized charge TX-{tx} for ${amt} on my statement, dispute it.", "Open a fraud chargeback for transaction TX-{tx} that I never authorized."],
    5: ["Set up a monthly automatic payment of ${amt} to my electric utility provider.", "Schedule a recurring bill pay of ${amt} every month for internet service."],
    6: ["Show all my transactions on account {acc} between 2026-01-01 and 2026-03-31.", "List my recent account history for {acc} over the past month."],
    7: ["Calculate the monthly payment on a ${amt}00 loan at 6.5% interest over 36 months.", "Give me a loan amortization breakdown for ${amt}00 over 48 months at 5.9%."],
    8: ["Create a disposable virtual card with a ${amt} limit valid for 7 days.", "Generate a single-use virtual card capped at ${amt} for an online purchase."],
    9: ["I am flying to Japan next week, register a travel notice so my card works there.", "Add an international travel alert for Spain from 2026-12-01 to 2026-12-15."],
    10: ["Download my 2025 annual tax withholding statement as a PDF.", "Export my interest tax summary document for fiscal year 2025."],
    11: ["Place a limit buy order for 25 shares of NVDA at ${amt}.", "Submit a limit order to sell 10 shares of AAPL at ${amt}."],
    12: ["What is the current stock price and trading volume for MSFT?", "Fetch the latest real-time market quote for ticker GOOGL."],
    13: ["Cancel my recurring monthly subscription with StreamPlus immediately.", "Stop future automatic billing from FitnessClub on my account."],
    14: ["Check if my passport KYC identity verification has been approved.", "Verify the compliance status of my uploaded driver license document."],
    15: ["Alert me if my dining and restaurants spending goes over ${amt} this month.", "Set a monthly budget notification at ${amt} for groceries."],
    16: ["Find the closest fee-free ATMs within 5 km of downtown Lima.", "Where is the nearest supported ATM around postal code 15046?"],
    17: ["Convert ${amt} from my USD balance into BTC.", "Swap 0.5 ETH into USDC in my crypto wallet."],
    18: ["Split a ${amt} dinner bill equally among Ana, Luis, and Marco.", "Send a shared expense request for ${amt} to my three roommates."],
    19: ["Redeem 15,000 cashback points into my main checking account {acc}.", "Convert my 25,000 loyalty reward points to statement credit."],
    20: ["Request a credit limit increase on card {card} to $12,000.", "Apply to raise my revolving credit limit on {card} given my new salary."],
    21: ["Add a new international wire beneficiary with IBAN DE89370400440532013000.", "Register an external SWIFT recipient for overseas transfers."],
    22: ["How much are the wire fees to send ${amt}0 to Germany in EUR?", "Estimate the international remittance cost for sending ${amt}0 to Mexico."],
    23: ["Check if I qualify for mortgage preapproval on a $280,000 home with $60,000 down.", "Evaluate my mortgage eligibility given $95,000 annual income and $45,000 down payment."],
}

_AMBIGUOUS_TEMPLATES: list[tuple[str, int, str]] = [
    ("A strange subscription charged my card TX-9921 and I want to get my money back and stop them.", 4, "security"),
    ("I am traveling to Europe tomorrow and need to know how much sending EUR costs or what the rate is.", 3, "fx"),
    ("My card was declined at an online store; maybe I need a higher limit or a temporary virtual card?", 8, "cards"),
    ("I want to invest some cash into tech—can you check the price of NVDA before I decide on an order?", 12, "investing"),
    ("Review my spending last quarter on account ACC-402 so I can prepare my tax paperwork.", 6, "banking"),
    ("I am buying an apartment and need to see what monthly payments look like for a 20-year loan.", 7, "loans"),
    ("Someone in another country needs money from me; first tell me the fee before I register their IBAN.", 22, "fx"),
    ("My card might have been compromised while I was abroad without a travel notice; lock it first.", 2, "security"),
]


def load_bfcl_benchmark(
    num_samples: int = 200,
    ambiguous_ratio: float = 0.30,
    seed: int = 42,
) -> BFCLBenchmarkData:
    """Build a deterministic BFCL function-calling benchmark dataset.

    Combines clear single-tool queries (for Level-1 fast dispatch) with ambiguous
    multi-intent queries (which induce lower max probability and trigger Level-2 ReAct).
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if not 0.0 <= ambiguous_ratio <= 1.0:
        raise ValueError("ambiguous_ratio must be in [0.0, 1.0]")

    rng = random.Random(seed)
    tools = list(DEFAULT_BFCL_TOOLS)
    num_ambiguous = int(round(num_samples * ambiguous_ratio))
    num_clear = num_samples - num_ambiguous

    samples: list[BFCLQuerySample] = []

    for i in range(num_clear):
        tool = tools[i % len(tools)]
        templates = _CLEAR_TEMPLATES[tool.tool_id]
        tpl = templates[(i // len(tools)) % len(templates)]
        acc = f"ACC-{1000 + i}"
        acc2 = f"ACC-{2000 + i}"
        card = f"CARD-{4000 + i}"
        curr = ["USD", "PEN", "EUR", "GBP"][i % 4]
        amt = str(50 + (i * 37) % 900)
        tx = str(80000 + i)
        query = tpl.format(acc=acc, acc2=acc2, card=card, curr=curr, amt=amt, tx=tx)
        samples.append(
            BFCLQuerySample(
                query=query,
                class_idx=tool.tool_id,
                is_ambiguous=False,
                required_arguments={"account_id": acc, "amount": amt},
                domain=tool.domain,
            )
        )

    for j in range(num_ambiguous):
        tpl_q, target_id, dom = _AMBIGUOUS_TEMPLATES[j % len(_AMBIGUOUS_TEMPLATES)]
        query = f"{tpl_q} (Case #{j + 1})"
        samples.append(
            BFCLQuerySample(
                query=query,
                class_idx=target_id,
                is_ambiguous=True,
                required_arguments={"case_id": str(j + 1)},
                domain=dom,
            )
        )

    rng.shuffle(samples)

    # Construct multi-question shared-state scenarios for Tree Batching evaluation
    shared_state = (
        "Customer Audit Dossier #CA-2026-8841:\n"
        "- Primary Checking Account: ACC-1042 (Balance: $14,250.00 USD, Status: Active)\n"
        "- Corporate Visa Card: CARD-7781 (Limit: $10,000, Active Travel Notice: None)\n"
        "- Flagged Event: Unrecognized charge TX-99412 at 'GlobalGaming Ltd' for $489.00 on 2026-11-18.\n"
        "- Pending Request: Customer plans travel to Tokyo (2026-12-05 to 2026-12-19), wants to swap "
        "$1,500 USD to JPY or check the live FX rate, and requested 2025 tax withholding statements.\n"
        "- Investment Portfolio: Holds 40 shares of NVDA and 25 shares of MSFT; recurring billing "
        "active for 'StreamPlus Monthly' ($19.99/mo) and 'CloudStorage Pro' ($9.99/mo).\n"
    )
    tree_questions = [
        ("Which tool should be invoked to contest the unrecognized $489.00 charge TX-99412?", 4),
        ("Which tool should be called first to protect CARD-7781 from further unauthorized use?", 2),
        ("Which tool registers the customer's upcoming December trip to Tokyo?", 9),
        ("Which tool checks the live conversion rate between USD and JPY?", 3),
        ("Which tool generates the customer's 2025 annual tax withholding PDF?", 10),
        ("Which tool cancels the recurring 'StreamPlus Monthly' billing?", 13),
        ("Which tool retrieves the current balance of checking account ACC-1042?", 0),
        ("Which tool fetches the real-time market price of the customer's NVDA holdings?", 12),
        ("Which tool lists all past transactions on ACC-1042 for November 2026?", 6),
        ("Which tool creates a temporary virtual card for safe online purchases?", 8),
        ("Which tool sets a monthly spending alert on the customer's dining category?", 15),
        ("Which tool submits a limit buy order for additional shares of MSFT?", 11),
        ("Which tool estimates the international wire fee to send funds to Japan?", 22),
        ("Which tool redeems the customer's accumulated cashback points?", 19),
        ("Which tool checks fee-free ATMs near the customer's hotel?", 16),
        ("Which tool applies for a credit limit increase on CARD-7781?", 20),
    ]

    tree_scenarios = [
        BFCLTreeScenario(
            state_text=shared_state,
            questions=[q for q, _ in tree_questions[:m]],
            class_idx_list=[cid for _, cid in tree_questions[:m]],
        )
        for m in (1, 2, 4, 8, 16)
    ]

    return BFCLBenchmarkData(
        tools=tools,
        samples=samples,
        tree_scenarios=tree_scenarios,
    )


def format_bfcl_decision_prompt(
    sample: BFCLQuerySample,
    tools: Sequence[BFCLToolSchema],
    shuffle_options: bool = False,
    rng: random.Random | None = None,
) -> DecisionExample:
    """Format a BFCL function-calling sample into a single-pass (T=1) decision prompt."""
    option_labels = [t.prompt_label for t in tools]
    return format_decision_prompt(
        query=sample.query,
        option_names=option_labels,
        class_idx=sample.class_idx,
        shuffle_options=shuffle_options,
        rng=rng,
        humanize_labels=False,
    )
