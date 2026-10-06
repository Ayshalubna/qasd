"""Which contact-centre team handles each of the 77 Banking77 intents, and how urgent it is."""
from __future__ import annotations

TEAMS = {
    "Fraud & security": ("urgent", ["compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone", "card_payment_not_recognised",
                                    "cash_withdrawal_not_recognised", "direct_debit_payment_not_recognised", "pin_blocked",
                                    "passcode_forgotten", "change_pin"]),
    "Cards": ("normal", ["activate_my_card", "card_about_to_expire", "card_acceptance", "card_arrival", "card_delivery_estimate",
                         "card_linking", "card_not_working", "card_swallowed", "contactless_not_working", "declined_card_payment",
                         "disposable_card_limits", "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
                         "getting_virtual_card", "order_physical_card", "virtual_card_not_working", "visa_or_mastercard",
                         "apple_pay_or_google_pay", "supported_cards_and_currencies", "pending_card_payment", "reverted_card_payment?"]),
    "Payments & transfers": ("normal", ["balance_not_updated_after_bank_transfer", "beneficiary_not_allowed", "cancel_transfer",
                                        "declined_transfer", "failed_transfer", "pending_transfer", "receiving_money",
                                        "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
                                        "transfer_fee_charged"]),
    "Top-ups": ("normal", ["automatic_top_up", "pending_top_up", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
                           "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
                           "verify_top_up", "balance_not_updated_after_cheque_or_cash_deposit"]),
    "Cash & ATM": ("normal", ["atm_support", "cash_withdrawal_charge", "declined_cash_withdrawal", "pending_cash_withdrawal",
                              "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"]),
    "FX & fees": ("normal", ["card_payment_fee_charged", "card_payment_wrong_exchange_rate", "exchange_charge", "exchange_rate",
                             "exchange_via_app", "extra_charge_on_statement", "fiat_currency_support"]),
    "Disputes & refunds": ("priority", ["Refund_not_showing_up", "request_refund", "transaction_charged_twice"]),
    "Account & identity": ("normal", ["age_limit", "edit_personal_details", "terminate_account", "unable_to_verify_identity",
                                      "verify_my_identity", "verify_source_of_funds", "why_verify_identity", "country_support"]),
}
ROUTE = {intent: (team, urgency) for team, (urgency, intents) in TEAMS.items() for intent in intents}


def pretty(intent: str) -> str:
    return intent.replace("_", " ").replace("?", "").strip().capitalize()
