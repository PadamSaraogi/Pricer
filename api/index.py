from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add backend directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from bs_core import OptionInput, bs_prices, bs_greeks, implied_vol, moneyness_tags
from american_binomial import crr_price_european, crr_price_american
from bonds import (
    price_bond,
    ytm_from_price,
    macaulay_duration,
    modified_duration,
    convexity_numeric,
    price_bond_dates,
    ytm_from_price_dates,
)
from yield_curve import YieldCurve
from swaps import par_swap_rate, swap_pv, dv01
from forwards import forward_price, forward_price_from_curve
from black76 import black76_price, black76_greeks, implied_vol_b76

app = Flask(__name__)
CORS(app)

@app.route("/api/python")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/pricing/european", methods=["POST"])
def european_pricing():
    data = request.json
    try:
        inp = OptionInput(
            S0=float(data["S0"]),
            K=float(data["K"]),
            r=float(data["r"]),
            sigma=float(data["sigma"]),
            T=float(data["T"]),
            q=float(data.get("q", 0.0))
        )
        call, put, d1, d2 = bs_prices(inp)
        greeks = bs_greeks(inp)
        tags = moneyness_tags(inp.S0, inp.K, d1)
        
        return jsonify({
            "call_price": call,
            "put_price": put,
            "d1": d1,
            "d2": d2,
            "greeks": greeks,
            "tags": tags
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/pricing/american", methods=["POST"])
def american_pricing():
    data = request.json
    try:
        inp = OptionInput(
            S0=float(data["S0"]),
            K=float(data["K"]),
            r=float(data["r"]),
            sigma=float(data["sigma"]),
            T=float(data["T"]),
            q=float(data.get("q", 0.0))
        )
        otype = data.get("option_type", "call").lower()
        steps = int(data.get("steps", 200))
        
        euro_tree = crr_price_european(inp, otype, steps)
        amer_tree = crr_price_american(inp, otype, steps)
        
        return jsonify({
            "euro_tree": euro_tree,
            "amer_tree": amer_tree,
            "steps": steps
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/bond/analytics", methods=["POST"])
def bond_analytics():
    data = request.json
    try:
        # Simple bond pricing or date-based
        if "settlement" in data and "maturity" in data:
            # Date-based
            face = float(data.get("face", 100))
            coupon = float(data.get("coupon", 0.05))
            freq = int(data.get("freq", 2))
            yld = float(data["yield"])
            settlement = data["settlement"]
            maturity = data["maturity"]
            dc = data.get("dc", "ACT/365F")
            
            price = price_bond_dates(settlement, maturity, coupon, yld, freq, face, dc)
            ytm = ytm_from_price_dates(settlement, maturity, coupon, price, freq, face, dc) or yld
            
            return jsonify({
                "price": price,
                "ytm": ytm
            })
        else:
            # Simple T-based
            face = float(data.get("face", 100))
            coupon = float(data.get("coupon", 0.05))
            T = float(data["T"])
            freq = int(data.get("freq", 2))
            yld = float(data["yield"])
            
            price = price_bond(T, coupon, yld, freq, face)
            ytm = ytm_from_price(T, coupon, price, freq, face) or yld
            mac_dur = macaulay_duration(T, coupon, yld, freq, face)
            mod_dur = modified_duration(T, coupon, yld, freq, face)
            conv = convexity_numeric(T, coupon, yld, freq, face)
            
            return jsonify({
                "price": price,
                "ytm": ytm,
                "macaulay_duration": mac_dur,
                "modified_duration": mod_dur,
                "convexity": conv
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/curve/bootstrap", methods=["POST"])
def bootstrap():
    data = request.json
    try:
        # data["deposits"] = [{"T": 0.5, "rate": 0.05}, ...]
        # data["bonds"] = [{"T": 1, "price": 100, "coupon": 0.05, "freq": 2}, ...]
        deposits = data.get("deposits", [])
        bonds = data.get("bonds", [])
        
        curve = YieldCurve(
            mats=[d["T"] for d in deposits],
            zeros=[d["rate"] for d in deposits],
            dfs=[1.0 / (1.0 + d["rate"])**d["T"] for d in deposits]
        )
        
        # If we have bonds, we should actually bootstrap, but for MVP
        # we'll return the data points for charting
        
        return jsonify({
            "maturities": curve.mats,
            "zeros": curve.zeros,
            "dfs": curve.dfs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
