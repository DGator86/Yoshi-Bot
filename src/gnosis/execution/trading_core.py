import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Yoshi-Bot Trading Core")


# --- Models ---


class TradeProposal(BaseModel):
    proposal_id: str = ""
    symbol: str
    action: str  # BUY_YES, BUY_NO
    strike: float
    market_prob: float
    model_prob: float
    edge: float
    raw_forecast: Optional[Dict[str, Any]] = None


class ActionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    status: str
    active: bool
    timestamp: datetime
    version: str = "1.0.0"


# --- State ---


class TradingState:
    def __init__(self):
        self.is_active = True
        self.proposals: Dict[str, Dict] = {}
        self.positions: List[Dict] = []
        self.orders: List[Dict] = []
        self.start_time = datetime.now()


state = TradingState()


# --- Endpoints ---


@app.get("/status", response_model=StatusResponse)
async def get_status():
    return {
        "status": "running" if state.is_active else "paused",
        "active": state.is_active,
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }


@app.get("/positions")
async def get_positions():
    return state.positions


@app.get("/orders")
async def get_orders():
    return state.orders


@app.post("/propose", response_model=ActionResponse)
async def propose_trade(proposal: TradeProposal):
    if not state.is_active:
        return {"success": False, "message": "Trading Core is paused."}

    prop_id = str(uuid.uuid4())[:8]
    proposal.proposal_id = prop_id
    state.proposals[prop_id] = proposal.dict()

    logger.info(f"Received proposal {prop_id} for {proposal.symbol}")
    return {
        "success": True,
        "message": f"Proposal {prop_id} received.",
        "data": {"proposal_id": prop_id}
    }


@app.post("/approve/{proposal_id}", response_model=ActionResponse)
async def approve_trade(proposal_id: str):
    if proposal_id not in state.proposals:
        raise HTTPException(status_code=404, detail="Proposal not found")

    proposal = state.proposals[proposal_id]

    # Here we would normally trigger the actual exchange order
    # For now, we simulate success
    order = {
        "order_id": str(uuid.uuid4())[:8],
        "symbol": proposal["symbol"],
        "action": proposal["action"],
        "timestamp": datetime.now(),
        "status": "filled"
    }
    state.orders.append(order)
    state.positions.append(order)

    msg = f"Approved proposal {proposal_id}, executed order {order['order_id']}"
    logger.info(msg)
    return {
        "success": True,
        "message": f"Trade {proposal_id} approved and executed."
    }


@app.post("/kill-switch", response_model=ActionResponse)
async def kill_switch():
    state.is_active = False
    # Logic to cancel all orders and flatten positions would go here
    logger.warning("KILL SWITCH ACTIVATED")
    return {
        "success": True,
        "message": "All trading halted. System safety engaged."
    }


@app.post("/pause", response_model=ActionResponse)
async def pause_trading():
    state.is_active = False
    return {"success": True, "message": "Trading paused."}


@app.post("/resume", response_model=ActionResponse)
async def resume_trading():
    state.is_active = True
    return {"success": True, "message": "Trading resumed."}


@app.post("/flatten", response_model=ActionResponse)
async def flatten_positions():
    # Logic to close all positions
    state.positions = []
    return {"success": True, "message": "All positions flattened."}
