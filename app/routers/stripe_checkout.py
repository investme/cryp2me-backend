"""
app/routers/stripe_checkout.py — cryp2me.ai Stripe Checkout

Creates Checkout Sessions and receives webhook events.
Designed so swapping test → live keys is just changing env vars
(STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET) and the price IDs below.

Required env vars on Render:
    STRIPE_SECRET_KEY        - sk_test_... or sk_live_...
    STRIPE_WEBHOOK_SECRET    - whsec_... (from Stripe dashboard webhook setup)
    STRIPE_PRICE_MONTHLY     - price_... for $39.99/month plan
    STRIPE_PRICE_YEARLY      - price_... for $335.99/year plan
    PUBLIC_SITE_URL          - e.g. https://cryp2me.ai (for redirect URLs)
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import stripe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["stripe"])

# ─── Config from env (fail loud if missing in production) ──────────────────
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
PRICE_MONTHLY  = os.getenv("STRIPE_PRICE_MONTHLY", "price_1TcbHVRsWRcsNjUuBLHCii7q")
PRICE_YEARLY   = os.getenv("STRIPE_PRICE_YEARLY",  "price_1TcbKPRsWRcsNjUuqM8Zx5B6")
PUBLIC_SITE    = os.getenv("PUBLIC_SITE_URL", "https://cryp2me.ai").rstrip("/")


@router.post("/create-checkout-session")
async def create_checkout_session(request: Request):
    """
    Body: { "plan": "monthly" | "yearly", "email": "optional@x.com" }
    Returns: { "url": "https://checkout.stripe.com/..." }
    Frontend redirects the user to this URL.
    """
    if not stripe.api_key:
        raise HTTPException(500, "Stripe not configured on the server")

    try:
        body = await request.json()
    except Exception:
        body = {}

    plan  = (body.get("plan") or "monthly").lower()
    email = body.get("email") or None

    price_id = PRICE_YEARLY if plan == "yearly" else PRICE_MONTHLY

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=email,
            success_url=f"{PUBLIC_SITE}/app?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{PUBLIC_SITE}/?checkout=cancel",
            allow_promotion_codes=True,
            billing_address_collection="auto",
        )
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating session: {e}")
        raise HTTPException(400, f"Stripe error: {e.user_message or str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating session: {e}")
        raise HTTPException(500, "Could not create checkout session")

    return JSONResponse({"url": session.url, "id": session.id})


@router.get("/verify-session/{session_id}")
async def verify_session(session_id: str):
    """
    Frontend calls this after returning from Stripe success_url
    to confirm the session is paid and grant Pro access locally.
    """
    if not stripe.api_key:
        raise HTTPException(500, "Stripe not configured")

    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except stripe.error.StripeError as e:
        raise HTTPException(400, f"Could not retrieve session: {e}")

    paid = session.payment_status == "paid"
    # For subscriptions, also check the subscription is active
    sub_active = False
    plan = None
    if session.subscription:
        try:
            sub = stripe.Subscription.retrieve(session.subscription)
            sub_active = sub.status in ("active", "trialing")
            # detect monthly vs yearly from price id
            if sub["items"]["data"]:
                pid = sub["items"]["data"][0]["price"]["id"]
                if pid == PRICE_YEARLY:
                    plan = "yearly"
                elif pid == PRICE_MONTHLY:
                    plan = "monthly"
        except Exception as e:
            logger.warning(f"Could not retrieve subscription: {e}")

    return {
        "paid":           paid,
        "subscription_active": sub_active,
        "plan":           plan,
        "customer_email": session.customer_details.email if session.customer_details else None,
    }


@router.post("/stripe-webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    """
    Receives events from Stripe. Configure this URL in Stripe dashboard:
    https://dashboard.stripe.com/test/webhooks
    Endpoint: https://cryp2me.ai/api/stripe-webhook
    Events to send:
      - checkout.session.completed
      - customer.subscription.deleted
      - customer.subscription.updated
      - invoice.payment_failed
    """
    if not WEBHOOK_SECRET:
        logger.warning("Webhook hit but STRIPE_WEBHOOK_SECRET not set — skipping verify")
        return {"received": True, "verified": False}

    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(400, "Invalid signature")

    event_type = event["type"]
    data = event["data"]["object"]

    # Right now we just log events. When you wire a real DB of subscribers,
    # update their status here. For now the frontend uses /verify-session to
    # decide Pro status immediately after checkout.
    if event_type == "checkout.session.completed":
        logger.info(f"✓ Checkout completed: {data.get('id')} email={data.get('customer_email')}")
    elif event_type == "customer.subscription.deleted":
        logger.info(f"✗ Subscription cancelled: {data.get('id')}")
    elif event_type == "invoice.payment_failed":
        logger.warning(f"⚠ Payment failed: {data.get('id')}")
    else:
        logger.info(f"Stripe event: {event_type}")

    return {"received": True}

@router.get("/stripe-diag")
async def stripe_diag():
    """TEMP: shows whether env vars are loaded. Remove before going live."""
    sk = os.getenv("STRIPE_SECRET_KEY", "")
    return {
        "secret_key_set":      bool(sk),
        "secret_key_prefix":   sk[:8] if sk else "",
        "secret_key_length":   len(sk),
        "secret_key_starts_with_sk": sk.startswith("sk_"),
        "price_monthly":       os.getenv("STRIPE_PRICE_MONTHLY", "")[:15] + "..." if os.getenv("STRIPE_PRICE_MONTHLY") else "MISSING",
        "price_yearly":        os.getenv("STRIPE_PRICE_YEARLY", "")[:15] + "..." if os.getenv("STRIPE_PRICE_YEARLY") else "MISSING",
        "public_site_url":     os.getenv("PUBLIC_SITE_URL", "MISSING"),
        "webhook_secret_set":  bool(os.getenv("STRIPE_WEBHOOK_SECRET", "")),
        "stripe_api_key_var":  bool(stripe.api_key),
    }
